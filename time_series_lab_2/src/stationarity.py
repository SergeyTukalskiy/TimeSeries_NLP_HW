import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')

class StationarityTransformer:
    """Преобразования для стационарности временных рядов"""
    
    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.original_series = series.copy()
        self.transformations = {}
        self.lambda_boxcox = None
    
    def log_transform(self) -> pd.Series:
        """Логарифмическое преобразование"""
        if (self.series <= 0).any():
            raise ValueError("Логарифмическое преобразование требует положительных значений")
        
        transformed = np.log(self.series)
        self.transformations['log'] = transformed
        return transformed
    
    def boxcox_transform(self, lmbda: float = None) -> Tuple[pd.Series, float]:
        """Преобразование Бокса-Кокса"""
        if (self.series <= 0).any():
            # Сдвиг для положительных значений
            shift = -self.series.min() + 0.01
            series_positive = self.series + shift
        else:
            series_positive = self.series
        
        if lmbda is None:
            lmbda = boxcox_normmax(series_positive)
        
        transformed, fitted_lambda = boxcox(series_positive, lmbda=lmbda)
        transformed_series = pd.Series(transformed, index=self.series.index)
        
        self.transformations['boxcox'] = transformed_series
        self.lambda_boxcox = fitted_lambda
        return transformed_series, fitted_lambda
    
    def difference(self, order: int = 1) -> pd.Series:
        """Дифференцирование первого порядка"""
        differentiated = self.series.diff(order).dropna()
        self.transformations[f'diff_{order}'] = differentiated
        return differentiated
    
    def seasonal_difference(self, seasonal_period: int = 7) -> pd.Series:
        """Сезонное дифференцирование"""
        seasonal_diff = self.series.diff(seasonal_period).dropna()
        self.transformations[f'seasonal_diff_{seasonal_period}'] = seasonal_diff
        return seasonal_diff
    
    def combined_difference(self, order: int = 1, seasonal_period: int = 7) -> pd.Series:
        """Комбинированное дифференцирование"""
        regular_diff = self.series.diff(order)
        combined_diff = regular_diff.diff(seasonal_period).dropna()
        self.transformations[f'combined_diff_{order}_{seasonal_period}'] = combined_diff
        return combined_diff
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Комплексный тест на стационарность"""
        # ADF тест
        adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(series.dropna())
        
        # KPSS тест
        try:
            kpss_stat, kpss_pvalue, _, kpss_critical = kpss(series.dropna(), regression='c')
        except:
            kpss_stat, kpss_pvalue, kpss_critical = np.nan, np.nan, {}
        
        # Дополнительные метрики
        mean = series.mean()
        std = series.std()
        autocorr_1 = series.autocorr(lag=1)
        
        return {
            'ADF': {
                'statistic': adf_stat,
                'pvalue': adf_pvalue,
                'stationary': adf_pvalue < 0.05
            },
            'KPSS': {
                'statistic': kpss_stat,
                'pvalue': kpss_pvalue,
                'stationary': kpss_pvalue > 0.05 if not np.isnan(kpss_pvalue) else False
            },
            'summary': {
                'is_stationary': adf_pvalue < 0.05 and (kpss_pvalue > 0.05 if not np.isnan(kpss_pvalue) else True),
                'mean': mean,
                'std': std,
                'autocorr_lag1': autocorr_1
            }
        }
    
    def find_optimal_transformation(self, methods: List[str] = None) -> Dict[str, Any]:
        """Поиск оптимального преобразования"""
        if methods is None:
            methods = ['original', 'log', 'boxcox', 'diff_1', 'seasonal_diff_7', 'combined_diff_1_7']
        
        results = {}
        
        for method in methods:
            try:
                if method == 'original':
                    series_test = self.series
                elif method == 'log':
                    series_test = self.log_transform()
                elif method == 'boxcox':
                    series_test, _ = self.boxcox_transform()
                elif method.startswith('diff_'):
                    order = int(method.split('_')[1])
                    series_test = self.difference(order)
                elif method.startswith('seasonal_diff_'):
                    period = int(method.split('_')[2])
                    series_test = self.seasonal_difference(period)
                elif method.startswith('combined_diff_'):
                    parts = method.split('_')
                    order = int(parts[2])
                    period = int(parts[3])
                    series_test = self.combined_difference(order, period)
                else:
                    continue
                
                stationarity_test = self.test_stationarity(series_test)
                results[method] = {
                    'series': series_test,
                    'stationarity': stationarity_test,
                    'score': self._calculate_stationarity_score(stationarity_test)
                }
                
            except Exception as e:
                print(f"Ошибка в методе {method}: {e}")
                continue
        
        # Выбор лучшего метода
        best_method = max(results.keys(), key=lambda x: results[x]['score'])
        
        return {
            'best_method': best_method,
            'best_series': results[best_method]['series'],
            'all_results': results,
            'comparison': {method: result['score'] for method, result in results.items()}
        }
    
    def _calculate_stationarity_score(self, stationarity_result: Dict) -> float:
        """Расчет скора стационарности"""
        adf_score = 1 if stationarity_result['ADF']['stationary'] else 0
        kpss_score = 1 if stationarity_result['KPSS']['stationary'] else 0
        std_score = 1 / (1 + stationarity_result['summary']['std'])  # Штраф за высокую волатильность
        
        total_score = adf_score * 0.4 + kpss_score * 0.4 + std_score * 0.2
        return total_score
    
    def inverse_transform(self, transformed_series: pd.Series, method: str) -> pd.Series:
        """Обратное преобразование"""
        if method == 'log':
            return np.exp(transformed_series)
        elif method == 'boxcox':
            from scipy.special import inv_boxcox
            return inv_boxcox(transformed_series, self.lambda_boxcox)
        elif method.startswith('diff_'):
            # Для дифференцирования нужен первый элемент исходного ряда
            order = int(method.split('_')[1])
            original_first = self.original_series.iloc[:order]
            restored = transformed_series.copy()
            for i in range(order):
                restored = restored.cumsum()
                restored.iloc[0] += original_first.iloc[i]
            return restored
        else:
            return transformed_series