import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

class ForecastingStrategies:
    """Реализация стратегий многопшагового прогнозирования"""
    
    def __init__(self, model_factory: Callable, target_column: str):
        self.model_factory = model_factory
        self.target_column = target_column
        self.results = {}
    
    def recursive_strategy(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                         h: int, features: List[str]) -> Dict[str, Any]:
        """Рекурсивная стратегия"""
        start_time = time.time()
        
        X_train = train_data[features]
        y_train = train_data[self.target_column]
        X_test = test_data[features].copy()
        
        model = self.model_factory()
        model.fit(X_train, y_train)
        
        predictions = []
        current_features = X_test.iloc[0:1].copy()
        
        for step in range(h):
            if step >= len(X_test):
                break
                
            # Прогноз на один шаг
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            
            # Обновление признаков для следующего шага
            if step < h - 1:
                current_features = X_test.iloc[step+1:step+2].copy()
                # Обновление лаговых признаков
                for lag in [1, 7, 14, 30]:
                    lag_col = f'lag_{lag}'
                    if lag_col in features:
                        if step + 1 - lag >= 0 and step + 1 - lag < len(predictions):
                            current_features[lag_col] = predictions[step + 1 - lag]
        
        execution_time = time.time() - start_time
        
        return {
            'predictions': predictions[:len(test_data)],
            'strategy': 'recursive',
            'execution_time': execution_time,
            'metrics': self._calculate_metrics(test_data[self.target_column].values[:len(predictions)], 
                                           predictions)
        }
    
    def direct_strategy(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      h: int, features: List[str]) -> Dict[str, Any]:
        """Прямая стратегия (отдельная модель для каждого горизонта)"""
        start_time = time.time()
        
        models = []
        predictions = []
        
        for step in range(1, h + 1):
            if step > len(test_data):
                break
                
            # Создание целевой переменной для горизонта step
            y_train_step = train_data[self.target_column].shift(-step).dropna()
            X_train_step = train_data.loc[y_train_step.index][features]
            y_train_step = y_train_step.loc[X_train_step.index]
            
            model = self.model_factory()
            model.fit(X_train_step, y_train_step)
            models.append(model)
            
            # Прогноз для текущего горизонта
            X_test_step = test_data.iloc[step-1:step][features]
            pred = model.predict(X_test_step)[0]
            predictions.append(pred)
        
        execution_time = time.time() - start_time
        
        return {
            'predictions': predictions[:len(test_data)],
            'strategy': 'direct',
            'execution_time': execution_time,
            'models_count': len(models),
            'metrics': self._calculate_metrics(test_data[self.target_column].values[:len(predictions)], 
                                           predictions)
        }
    
    def hybrid_strategy(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      h: int, features: List[str], split_point: int = None) -> Dict[str, Any]:
        """Гибридная стратегия"""
        if split_point is None:
            split_point = h // 2
        
        start_time = time.time()
        
        # Рекурсивная стратегия для ближайших шагов
        recursive_result = self.recursive_strategy(train_data, test_data.iloc[:split_point], 
                                                split_point, features)
        
        # Прямая стратегия для дальних шагов
        direct_result = self.direct_strategy(train_data, test_data.iloc[split_point:], 
                                          h - split_point, features)
        
        # Объединение прогнозов
        combined_predictions = recursive_result['predictions'] + direct_result['predictions']
        combined_time = recursive_result['execution_time'] + direct_result['execution_time']
        
        execution_time = time.time() - start_time
        
        return {
            'predictions': combined_predictions[:len(test_data)],
            'strategy': 'hybrid',
            'execution_time': execution_time,
            'split_point': split_point,
            'metrics': self._calculate_metrics(test_data[self.target_column].values[:len(combined_predictions)], 
                                           combined_predictions)
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: List[float]) -> Dict[str, float]:
        """Расчет метрик качества"""
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def compare_strategies(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                         h: int, features: List[str]) -> pd.DataFrame:
        """Сравнение всех стратегий"""
        strategies = {
            'recursive': self.recursive_strategy,
            'direct': self.direct_strategy,
            'hybrid': self.hybrid_strategy
        }
        
        comparison_results = []
        
        for name, strategy_func in strategies.items():
            try:
                result = strategy_func(train_data, test_data, h, features)
                result['strategy_name'] = name
                comparison_results.append(result)
            except Exception as e:
                print(f"Ошибка в стратегии {name}: {e}")
        
        # Создание DataFrame для сравнения
        comparison_df = pd.DataFrame([{
            'Strategy': r['strategy_name'],
            'MAE': r['metrics']['MAE'],
            'RMSE': r['metrics']['RMSE'],
            'MAPE': r['metrics']['MAPE'],
            'Execution_Time': r['execution_time'],
            'Predictions': r['predictions']
        } for r in comparison_results])
        
        self.results = {r['strategy_name']: r for r in comparison_results}
        return comparison_df