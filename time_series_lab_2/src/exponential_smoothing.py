import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ExponentialSmoothingModels:
    """Модели экспоненциального сглаживания"""
    
    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.models = {}
        self.forecasts = {}
    
    def simple_exponential_smoothing(self, optimized: bool = True, 
                                   forecast_horizon: int = 30) -> Dict[str, Any]:
        """Простое экспоненциальное сглаживание"""
        try:
            model = ExponentialSmoothing(self.series, trend=None, seasonal=None)
            fitted_model = model.fit(optimized=optimized)
            
            # Прогноз
            forecast = fitted_model.forecast(forecast_horizon)
            
            # Доверительные интервалы
            if hasattr(fitted_model, 'prediction_intervals'):
                pred_int = fitted_model.prediction_intervals(forecast_horizon)
            else:
                # Эмпирические интервалы
                std = fitted_model.resid.std()
                pred_int = pd.DataFrame({
                    'lower': forecast - 1.96 * std,
                    'upper': forecast + 1.96 * std
                })
            
            result = {
                'model': fitted_model,
                'forecast': forecast,
                'confidence_intervals': pred_int,
                'residuals': fitted_model.resid,
                'params': fitted_model.params
            }
            
            self.models['SES'] = result
            return result
            
        except Exception as e:
            print(f"Ошибка в SES: {e}")
            return None
    
    def holt_additive(self, optimized: bool = True, 
                     forecast_horizon: int = 30) -> Dict[str, Any]:
        """Модель Хольта (аддитивный тренд)"""
        try:
            model = ExponentialSmoothing(self.series, trend='add', seasonal=None)
            fitted_model = model.fit(optimized=optimized)
            
            forecast = fitted_model.forecast(forecast_horizon)
            std = fitted_model.resid.std()
            pred_int = pd.DataFrame({
                'lower': forecast - 1.96 * std,
                'upper': forecast + 1.96 * std
            })
            
            result = {
                'model': fitted_model,
                'forecast': forecast,
                'confidence_intervals': pred_int,
                'residuals': fitted_model.resid,
                'params': fitted_model.params
            }
            
            self.models['Holt_Additive'] = result
            return result
            
        except Exception as e:
            print(f"Ошибка в Хольте (аддитивный): {e}")
            return None
    
    def holt_multiplicative(self, optimized: bool = True,
                           forecast_horizon: int = 30) -> Dict[str, Any]:
        """Модель Хольта (мультипликативный тренд)"""
        if (self.series <= 0).any():
            print("Мультипликативная модель требует положительных значений")
            return None
        
        try:
            model = ExponentialSmoothing(self.series, trend='mul', seasonal=None)
            fitted_model = model.fit(optimized=optimized)
            
            forecast = fitted_model.forecast(forecast_horizon)
            std = fitted_model.resid.std()
            pred_int = pd.DataFrame({
                'lower': forecast - 1.96 * std,
                'upper': forecast + 1.96 * std
            })
            
            result = {
                'model': fitted_model,
                'forecast': forecast,
                'confidence_intervals': pred_int,
                'residuals': fitted_model.resid,
                'params': fitted_model.params
            }
            
            self.models['Holt_Multiplicative'] = result
            return result
            
        except Exception as e:
            print(f"Ошибка в Хольте (мультипликативный): {e}")
            return None
    
    def naive_forecast(self, forecast_horizon: int = 30) -> Dict[str, Any]:
        """Наивный прогноз"""
        last_value = self.series.iloc[-1]
        forecast = pd.Series([last_value] * forecast_horizon)
        
        result = {
            'forecast': forecast,
            'model': 'naive',
            'residuals': pd.Series([0] * len(self.series))  # Фиктивные остатки
        }
        
        self.models['Naive'] = result
        return result
    
    def diagnose_model(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Диагностика адекватности модели"""
        residuals = model_result['residuals'].dropna()
        
        # Тест Льюнга-Бокса на автокорреляцию
        try:
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_statistic = lb_test['lb_stat'].iloc[0]
            lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        except:
            lb_statistic, lb_pvalue = np.nan, np.nan
        
        # Тесты на нормальность
        try:
            shapiro_test = shapiro(residuals)
            shapiro_stat, shapiro_pvalue = shapiro_test[0], shapiro_test[1]
        except:
            shapiro_stat, shapiro_pvalue = np.nan, np.nan
        
        try:
            normality_test = normaltest(residuals)
            normaltest_stat, normaltest_pvalue = normality_test[0], normality_test[1]
        except:
            normaltest_stat, normaltest_pvalue = np.nan, np.nan
        
        # Гомоскедастичность (визуальная оценка)
        residual_std = residuals.std()
        mean_abs_residual = np.mean(np.abs(residuals))
        
        return {
            'ljung_box': {
                'statistic': lb_statistic,
                'pvalue': lb_pvalue,
                'no_autocorr': lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else False
            },
            'normality': {
                'shapiro_stat': shapiro_stat,
                'shapiro_pvalue': shapiro_pvalue,
                'normaltest_stat': normaltest_stat,
                'normaltest_pvalue': normaltest_pvalue,
                'is_normal': shapiro_pvalue > 0.05 if not np.isnan(shapiro_pvalue) else False
            },
            'homoscedasticity': {
                'residual_std': residual_std,
                'mean_abs_residual': mean_abs_residual,
                'cv_residuals': residual_std / mean_abs_residual if mean_abs_residual != 0 else np.inf
            },
            'residuals_summary': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis()
            }
        }
    
    def compare_models(self, forecast_horizon: int = 30, 
                     test_data: pd.Series = None) -> pd.DataFrame:
        """Сравнение всех моделей"""
        models_to_fit = {
            'SES': self.simple_exponential_smoothing,
            'Holt_Additive': self.holt_additive,
            'Holt_Multiplicative': self.holt_multiplicative,
            'Naive': self.naive_forecast
        }
        
        comparison_results = []
        
        for name, model_func in models_to_fit.items():
            try:
                result = model_func(forecast_horizon=forecast_horizon)
                if result is None:
                    continue
                
                # Диагностика
                diagnosis = self.diagnose_model(result)
                
                # Расчет метрик на тестовых данных
                if test_data is not None:
                    metrics = self._calculate_test_metrics(result['forecast'], test_data)
                else:
                    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
                
                comparison_results.append({
                    'Model': name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'AIC': result['model'].aic if hasattr(result['model'], 'aic') else np.nan,
                    'BIC': result['model'].bic if hasattr(result['model'], 'bic') else np.nan,
                    'Residuals_Std': diagnosis['residuals_summary']['std'],
                    'Ljung_Box_pvalue': diagnosis['ljung_box']['pvalue'],
                    'Shapiro_pvalue': diagnosis['normality']['shapiro_pvalue'],
                    'Is_Optimal': diagnosis['ljung_box']['no_autocorr'] and diagnosis['normality']['is_normal']
                })
                
            except Exception as e:
                print(f"Ошибка при сравнении модели {name}: {e}")
                continue
        
        return pd.DataFrame(comparison_results)
    
    def _calculate_test_metrics(self, forecast: pd.Series, test_data: pd.Series) -> Dict[str, float]:
        """Расчет метрик на тестовых данных"""
        min_len = min(len(forecast), len(test_data))
        forecast = forecast.iloc[:min_len]
        test_data = test_data.iloc[:min_len]
        
        mae = np.mean(np.abs(test_data - forecast))
        rmse = np.sqrt(np.mean((test_data - forecast) ** 2))
        
        # MAPE с защитой от деления на ноль
        if (test_data != 0).all():
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        else:
            mape = np.nan
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def forecast_with_confidence(self, model_type: str = 'Holt_Additive', 
                               forecast_horizon: int = 30, 
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """Прогноз с доверительными интервалами"""
        try:
            if model_type == 'SES':
                result = self.simple_exponential_smoothing(forecast_horizon=forecast_horizon)
            elif model_type == 'Holt_Additive':
                result = self.holt_additive(forecast_horizon=forecast_horizon)
            elif model_type == 'Holt_Multiplicative':
                result = self.holt_multiplicative(forecast_horizon=forecast_horizon)
            elif model_type == 'Naive':
                result = self.naive_forecast(forecast_horizon=forecast_horizon)
            else:
                print(f"Неизвестный тип модели: {model_type}")
                return None
            
            if result is None:
                return None
            
            # Расчет доверительных интервалов
            residuals = result['residuals'].dropna()
            if len(residuals) > 0:
                std_error = residuals.std()
            else:
                # Для наивной модели используем стандартное отклонение ряда
                std_error = self.series.std()
            
            # Z-score для доверительного уровня
            from scipy import stats
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            
            forecast = result['forecast']
            lower_bound = forecast - z_score * std_error
            upper_bound = forecast + z_score * std_error
            
            result.update({
                'confidence_intervals': pd.DataFrame({
                    'lower': lower_bound,
                    'upper': upper_bound
                }),
                'confidence_level': confidence_level,
                'std_error': std_error
            })
            
            return result
            
        except Exception as e:
            print(f"Ошибка прогноза с доверительными интервалами: {e}")
            return None
    
    def export_forecast_results(self, model_type: str = 'Holt_Additive',
                              forecast_horizon: int = 30,
                              include_parameters: bool = True) -> Dict[str, Any]:
        """Экспорт результатов прогноза"""
        result = self.forecast_with_confidence(model_type, forecast_horizon)
        
        if result is None:
            return {}
        
        # Создаем даты прогноза
        forecast_dates = pd.date_range(
            start=self.series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        export_data = {
            'model_type': model_type,
            'forecast_horizon': forecast_horizon,
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_values': [float(x) for x in result['forecast'].tolist()],
            'confidence_intervals': {
                'lower': [float(x) for x in result['confidence_intervals']['lower'].tolist()],
                'upper': [float(x) for x in result['confidence_intervals']['upper'].tolist()]
            },
            'last_training_date': self.series.index[-1].strftime('%Y-%m-%d'),
            'generation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'series_info': {
                'length': len(self.series),
                'mean': float(self.series.mean()),
                'std': float(self.series.std()),
                'min': float(self.series.min()),
                'max': float(self.series.max())
            }
        }
        
        if include_parameters and 'params' in result:
            # Безопасное преобразование параметров
            params = result['params']
            export_data['model_parameters'] = self._safe_convert_params(params)
        
        # Добавляем статистики модели
        if 'residuals' in result:
            residuals = result['residuals'].dropna()
            if len(residuals) > 0:
                export_data['model_statistics'] = {
                    'residuals_mean': float(residuals.mean()),
                    'residuals_std': float(residuals.std()),
                    'residuals_mae': float(np.mean(np.abs(residuals))),
                    'aic': float(result['model'].aic) if hasattr(result['model'], 'aic') else None,
                    'bic': float(result['model'].bic) if hasattr(result['model'], 'bic') else None
                }
            else:
                export_data['model_statistics'] = {
                    'residuals_mean': 0.0,
                    'residuals_std': 0.0,
                    'residuals_mae': 0.0,
                    'aic': None,
                    'bic': None
                }
        
        return export_data
    
    def _safe_convert_params(self, params) -> Dict[str, Any]:
        """Безопасное преобразование параметров модели"""
        converted_params = {}
        
        if hasattr(params, 'items'):
            for key, value in params.items():
                try:
                    # Пробуем преобразовать в float
                    if hasattr(value, '__len__') and len(value) == 1:
                        converted_params[str(key)] = float(value[0])
                    else:
                        converted_params[str(key)] = float(value)
                except (TypeError, ValueError):
                    # Если не получается, сохраняем как строку
                    converted_params[str(key)] = str(value)
        else:
            # Если params не словарь, пытаемся обработать как массив
            try:
                if hasattr(params, '__len__'):
                    for i, value in enumerate(params):
                        try:
                            converted_params[f'param_{i}'] = float(value)
                        except (TypeError, ValueError):
                            converted_params[f'param_{i}'] = str(value)
                else:
                    converted_params['param_0'] = float(params)
            except (TypeError, ValueError):
                converted_params['param_0'] = str(params)
        
        return converted_params
    
    def export_to_dataframe(self, model_type: str = 'Holt_Additive',
                          forecast_horizon: int = 30) -> pd.DataFrame:
        """Экспорт прогноза в DataFrame"""
        result = self.forecast_with_confidence(model_type, forecast_horizon)
        
        if result is None:
            return pd.DataFrame()
        
        forecast_dates = pd.date_range(
            start=self.series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': result['forecast'],
            'lower_bound': result['confidence_intervals']['lower'],
            'upper_bound': result['confidence_intervals']['upper'],
            'confidence_level': result['confidence_level']
        })
        
        df.set_index('date', inplace=True)
        return df
    
    def get_model_summary(self, model_type: str = 'Holt_Additive') -> Dict[str, Any]:
        """Получение сводки по модели"""
        if model_type not in self.models:
            # Пытаемся создать модель
            if model_type == 'SES':
                self.simple_exponential_smoothing()
            elif model_type == 'Holt_Additive':
                self.holt_additive()
            elif model_type == 'Holt_Multiplicative':
                self.holt_multiplicative()
            elif model_type == 'Naive':
                self.naive_forecast()
        
        if model_type in self.models:
            model_result = self.models[model_type]
            diagnosis = self.diagnose_model(model_result)
            
            summary = {
                'model_type': model_type,
                'fitted': True,
                'parameters': model_result.get('params', {}),
                'diagnosis': diagnosis,
                'forecast_available': 'forecast' in model_result
            }
            
            if 'forecast' in model_result:
                summary['forecast_stats'] = {
                    'mean': float(model_result['forecast'].mean()),
                    'std': float(model_result['forecast'].std()),
                    'min': float(model_result['forecast'].min()),
                    'max': float(model_result['forecast'].max())
                }
            
            return summary
        else:
            return {
                'model_type': model_type,
                'fitted': False,
                'error': 'Модель не была обучена'
            }
    
    def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        available = []
        model_configs = [
            ('SES', self.simple_exponential_smoothing),
            ('Holt_Additive', self.holt_additive),
            ('Holt_Multiplicative', self.holt_multiplicative),
            ('Naive', self.naive_forecast)
        ]
        
        for name, model_func in model_configs:
            try:
                result = model_func(forecast_horizon=1)  # Тестируем на коротком горизонте
                if result is not None:
                    available.append(name)
            except:
                continue
        
        return available