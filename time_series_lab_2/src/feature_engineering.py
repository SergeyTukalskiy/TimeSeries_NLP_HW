import pandas as pd
import numpy as np
from typing import List, Dict, Any

class FeatureEngineer:
    """Создание признаков для временных рядов"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features_df = None
    
    def create_temporal_features(self, date_column: str = None) -> 'FeatureEngineer':
        """Создание временных признаков"""
        if date_column:
            dates = pd.to_datetime(self.df[date_column])
        else:
            dates = self.df.index
        
        # Базовые временные признаки
        self.df['day_of_week'] = dates.dayofweek
        self.df['month'] = dates.month
        self.df['quarter'] = dates.quarter
        self.df['year'] = dates.year
        self.df['day_of_year'] = dates.dayofyear
        self.df['week_of_year'] = dates.isocalendar().week.astype(int)
        
        # Циклические признаки
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        return self
    
    def create_lag_features(self, target_column: str, lags: List[int] = [1, 7, 14, 30]) -> 'FeatureEngineer':
        """Создание лаговых признаков"""
        for lag in lags:
            self.df[f'lag_{lag}'] = self.df[target_column].shift(lag)
        
        return self
    
    def create_rolling_features(self, target_column: str, windows: List[int] = [7, 30, 90]) -> 'FeatureEngineer':
        """Создание скользящих статистик"""
        for window in windows:
            self.df[f'rolling_mean_{window}'] = self.df[target_column].rolling(window=window).mean()
            self.df[f'rolling_std_{window}'] = self.df[target_column].rolling(window=window).std()
            self.df[f'rolling_min_{window}'] = self.df[target_column].rolling(window=window).min()
            self.df[f'rolling_max_{window}'] = self.df[target_column].rolling(window=window).max()
            
            # Коэффициент вариации с защитой от деления на ноль
            rolling_mean = self.df[f'rolling_mean_{window}']
            rolling_std = self.df[f'rolling_std_{window}']
            self.df[f'rolling_var_{window}'] = np.where(
                rolling_mean != 0, 
                rolling_std / rolling_mean, 
                0
            )
        
        return self
    
    def create_volatility_features(self, target_column: str, windows: List[int] = [7, 30]) -> 'FeatureEngineer':
        """Признаки волатильности"""
        returns = self.df[target_column].pct_change()
        
        for window in windows:
            self.df[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # Коэффициент вариации с защитой
            rolling_mean_col = f'rolling_mean_{window}'
            if rolling_mean_col in self.df.columns:
                self.df[f'cv_{window}'] = np.where(
                    self.df[rolling_mean_col] != 0,
                    self.df[f'rolling_std_{window}'] / self.df[rolling_mean_col],
                    0
                )
        
        return self
    
    def create_all_features(self, target_column: str, 
                          lags: List[int] = [1, 7, 14, 30],
                          windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """Создание всех признаков"""
        (self.create_temporal_features()
           .create_lag_features(target_column, lags)
           .create_rolling_features(target_column, windows)
           .create_volatility_features(target_column, windows))
        
        # Удаляем строки с NaN значениями, которые появились из-за лагов и скользящих окон
        self.features_df = self.df.copy()
        return self.features_df
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Категоризация признаков"""
        if self.features_df is None:
            return {}
        
        features = self.features_df.columns.tolist()
        
        # Сначала определим все категории
        temporal_features = [f for f in features if any(x in f for x in 
                          ['day_of_week', 'month', 'quarter', 'year', '_sin', '_cos', 'week_of_year'])]
        
        lag_features = [f for f in features if f.startswith('lag_')]
        rolling_features = [f for f in features if f.startswith('rolling_')]
        volatility_features = [f for f in features if f.startswith('volatility_') or f.startswith('cv_')]
        
        # Все остальные признаки (исключая целевую переменную и временные метки)
        other_features = [f for f in features if f not in temporal_features + lag_features + rolling_features + volatility_features]
        
        # Убираем из other_features временные метки и целевую переменную, если она есть
        timestamp_indicators = ['timestamp', 'date', 'time']
        other_features = [f for f in other_features if not any(indicator in f.lower() for indicator in timestamp_indicators)]
        
        categories = {
            'temporal': temporal_features,
            'lags': lag_features,
            'rolling': rolling_features,
            'volatility': volatility_features,
            'other': other_features
        }
        
        return categories
    
    def get_features_info(self) -> Dict[str, Any]:
        """Получение информации о признаках"""
        if self.features_df is None:
            return {}
        
        categories = self.get_feature_categories()
        info = {
            'total_features': len(self.features_df.columns),
            'categories_count': {category: len(features) for category, features in categories.items()},
            'features_with_missing': self.features_df.isnull().sum().to_dict(),
            'features_dtypes': self.features_df.dtypes.astype(str).to_dict()
        }
        
        return info
    
    def clean_features(self, max_missing_ratio: float = 0.5) -> pd.DataFrame:
        """Очистка признаков от пропусков и маловажных"""
        if self.features_df is None:
            return pd.DataFrame()
        
        df_clean = self.features_df.copy()
        
        # Удаляем признаки с большим количеством пропусков
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        features_to_drop = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
        df_clean = df_clean.drop(columns=features_to_drop)
        
        # Заполняем оставшиеся пропуски
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_clean