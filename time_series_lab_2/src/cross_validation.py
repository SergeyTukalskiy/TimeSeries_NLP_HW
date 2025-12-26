import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Any
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TimeSeriesCV:
    """Кросс-валидация для временных рядов"""
    
    def __init__(self, model_factory: Callable, target_column: str):
        self.model_factory = model_factory
        self.target_column = target_column
        self.cv_results = {}
    
    def rolling_window_cv(self, data: pd.DataFrame, features: List[str], 
                        train_size: int, test_size: int, step: int = 1) -> Dict[str, Any]:
        """Скользящее окно с фиксированной длиной обучения"""
        n_splits = (len(data) - train_size - test_size) // step + 1
        
        fold_metrics = []
        fold_predictions = []
        
        for i in range(n_splits):
            start_train = i * step
            end_train = start_train + train_size
            end_test = end_train + test_size
            
            if end_test > len(data):
                break
            
            train_fold = data.iloc[start_train:end_train]
            test_fold = data.iloc[end_train:end_test]
            
            # Обучение и прогноз
            model = self.model_factory()
            model.fit(train_fold[features], train_fold[self.target_column])
            predictions = model.predict(test_fold[features])
            
            # Расчет метрик
            metrics = self._calculate_fold_metrics(test_fold[self.target_column].values, predictions)
            metrics['fold'] = i + 1
            metrics['train_start'] = train_fold.index[0]
            metrics['train_end'] = train_fold.index[-1]
            metrics['test_start'] = test_fold.index[0]
            metrics['test_end'] = test_fold.index[-1]
            
            fold_metrics.append(metrics)
            fold_predictions.append({
                'fold': i + 1,
                'true': test_fold[self.target_column].values,
                'pred': predictions,
                'dates': test_fold.index
            })
        
        return {
            'metrics_df': pd.DataFrame(fold_metrics),
            'predictions': fold_predictions,
            'method': 'rolling_window',
            'train_size': train_size,
            'test_size': test_size,
            'step': step
        }
    
    def expanding_window_cv(self, data: pd.DataFrame, features: List[str],
                          initial_train_size: int, test_size: int, step: int = 1) -> Dict[str, Any]:
        """Расширяющееся окно"""
        n_splits = (len(data) - initial_train_size) // test_size
        
        fold_metrics = []
        fold_predictions = []
        
        for i in range(n_splits):
            end_train = initial_train_size + i * step
            end_test = end_train + test_size
            
            if end_test > len(data):
                break
            
            train_fold = data.iloc[:end_train]
            test_fold = data.iloc[end_train:end_test]
            
            # Обучение и прогноз
            model = self.model_factory()
            model.fit(train_fold[features], train_fold[self.target_column])
            predictions = model.predict(test_fold[features])
            
            # Расчет метрик
            metrics = self._calculate_fold_metrics(test_fold[self.target_column].values, predictions)
            metrics['fold'] = i + 1
            metrics['train_size'] = len(train_fold)
            metrics['test_start'] = test_fold.index[0]
            metrics['test_end'] = test_fold.index[-1]
            
            fold_metrics.append(metrics)
            fold_predictions.append({
                'fold': i + 1,
                'true': test_fold[self.target_column].values,
                'pred': predictions,
                'dates': test_fold.index
            })
        
        return {
            'metrics_df': pd.DataFrame(fold_metrics),
            'predictions': fold_predictions,
            'method': 'expanding_window',
            'initial_train_size': initial_train_size,
            'test_size': test_size,
            'step': step
        }
    
    def timeseries_split_cv(self, data: pd.DataFrame, features: List[str],
                          n_splits: int = 5, test_size: int = None) -> Dict[str, Any]:
        """TimeSeriesSplit из sklearn"""
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        fold_metrics = []
        fold_predictions = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
            train_fold = data.iloc[train_idx]
            test_fold = data.iloc[test_idx]
            
            # Обучение и прогноз
            model = self.model_factory()
            model.fit(train_fold[features], train_fold[self.target_column])
            predictions = model.predict(test_fold[features])
            
            # Расчет метрик
            metrics = self._calculate_fold_metrics(test_fold[self.target_column].values, predictions)
            metrics['fold'] = fold
            metrics['train_size'] = len(train_fold)
            metrics['test_size'] = len(test_fold)
            
            fold_metrics.append(metrics)
            fold_predictions.append({
                'fold': fold,
                'true': test_fold[self.target_column].values,
                'pred': predictions,
                'dates': test_fold.index
            })
        
        return {
            'metrics_df': pd.DataFrame(fold_metrics),
            'predictions': fold_predictions,
            'method': 'timeseries_split',
            'n_splits': n_splits,
            'test_size': test_size
        }
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет метрик для фолда"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def compare_cv_methods(self, data: pd.DataFrame, features: List[str],
                         methods: List[str] = ['rolling', 'expanding', 'ts_split'],
                         **kwargs) -> pd.DataFrame:
        """Сравнение методов кросс-валидации"""
        comparison_results = []
        
        if 'rolling' in methods:
            rolling_result = self.rolling_window_cv(
                data, features, 
                kwargs.get('train_size', 100),
                kwargs.get('test_size', 30),
                kwargs.get('step', 1)
            )
            comparison_results.append({
                'method': 'Rolling Window',
                'mean_MAE': rolling_result['metrics_df']['MAE'].mean(),
                'mean_RMSE': rolling_result['metrics_df']['RMSE'].mean(),
                'mean_MAPE': rolling_result['metrics_df']['MAPE'].mean(),
                'std_MAE': rolling_result['metrics_df']['MAE'].std(),
                'n_folds': len(rolling_result['metrics_df'])
            })
        
        if 'expanding' in methods:
            expanding_result = self.expanding_window_cv(
                data, features,
                kwargs.get('initial_train_size', 100),
                kwargs.get('test_size', 30),
                kwargs.get('step', 1)
            )
            comparison_results.append({
                'method': 'Expanding Window',
                'mean_MAE': expanding_result['metrics_df']['MAE'].mean(),
                'mean_RMSE': expanding_result['metrics_df']['RMSE'].mean(),
                'mean_MAPE': expanding_result['metrics_df']['MAPE'].mean(),
                'std_MAE': expanding_result['metrics_df']['MAE'].std(),
                'n_folds': len(expanding_result['metrics_df'])
            })
        
        if 'ts_split' in methods:
            ts_split_result = self.timeseries_split_cv(
                data, features,
                kwargs.get('n_splits', 5),
                kwargs.get('test_size', None)
            )
            comparison_results.append({
                'method': 'TimeSeries Split',
                'mean_MAE': ts_split_result['metrics_df']['MAE'].mean(),
                'mean_RMSE': ts_split_result['metrics_df']['RMSE'].mean(),
                'mean_MAPE': ts_split_result['metrics_df']['MAPE'].mean(),
                'std_MAE': ts_split_result['metrics_df']['MAE'].std(),
                'n_folds': len(ts_split_result['metrics_df'])
            })
        
        return pd.DataFrame(comparison_results)