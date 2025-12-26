import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesUtils:
    """Утилиты для работы с временными рядами"""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Загрузка и подготовка данных"""
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Расчет метрик качества"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    @staticmethod
    def plot_decomposition(decomposition_result, title: str = "Декомпозиция временного ряда"):
        """Визуализация декомпозиции"""
        # Исправленная версия для новой Plotly
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'],
            vertical_spacing=0.05
        )
        
        # Исходный ряд
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.observed.index,
                y=decomposition_result.observed.values,
                name='Исходный',
                line=dict(color='blue')
            ), row=1, col=1
        )
        
        # Тренд
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.trend.index,
                y=decomposition_result.trend.values,
                name='Тренд',
                line=dict(color='red')
            ), row=2, col=1
        )
        
        # Сезонность
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.seasonal.index,
                y=decomposition_result.seasonal.values,
                name='Сезонность',
                line=dict(color='green')
            ), row=3, col=1
        )
        
        # Остатки
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.resid.index,
                y=decomposition_result.resid.values,
                name='Остатки',
                line=dict(color='orange')
            ), row=4, col=1
        )
        
        # Обновление layout для общей оси X
        fig.update_xaxes(matches='x')
        fig.update_layout(
            height=800, 
            title_text=title,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_forecast_comparison(historical_data: pd.Series, forecasts: Dict[str, pd.Series], 
                               title: str = "Сравнение прогнозов"):
        """Визуализация сравнения прогнозов"""
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            name='Исторические данные',
            line=dict(color='black', width=2)
        ))
        
        # Прогнозы
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            if i < len(colors):
                color = colors[i]
            else:
                color = 'gray'
                
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                name=f'{model_name} прогноз',
                line=dict(color=color, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Дата",
            yaxis_title="Значение",
            height=500
        )
        
        return fig
        
    @staticmethod
    def generate_forecast_report(forecast_data: Dict, model_info: Dict) -> str:
        """Генерация текстового отчета о прогнозе"""
        report = f"""
    # Отчет о прогнозировании временного ряда

    ## Информация о модели
    - **Тип модели**: {model_info.get('model_type', 'N/A')}
    - **Дата обучения**: {model_info.get('training_date', 'N/A')}
    - **Горизонт прогноза**: {model_info.get('forecast_horizon', 'N/A')} дней

    ## Статистики модели
    - **AIC**: {model_info.get('aic', 'N/A'):.2f}
    - **BIC**: {model_info.get('bic', 'N/A'):.2f}
    - **Средняя абсолютная ошибка**: {model_info.get('mae', 'N/A'):.4f}

    ## Прогнозные значения
    """
    
        if 'forecast_values' in forecast_data:
            forecast_values = forecast_data['forecast_values']
            report += f"- **Количество прогнозов**: {len(forecast_values)}\n"
            report += f"- **Среднее прогнозное значение**: {np.mean(forecast_values):.2f}\n"
            report += f"- **Минимальное значение**: {np.min(forecast_values):.2f}\n"
            report += f"- **Максимальное значение**: {np.max(forecast_values):.2f}\n"
    
        report += f"\n*Отчет сгенерирован: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
        return report