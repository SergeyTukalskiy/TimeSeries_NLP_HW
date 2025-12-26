import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from datetime import datetime

# Добавление пути к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from decomposition import DecompositionAnalyzer
from feature_engineering import FeatureEngineer
from forecasting_strategies import ForecastingStrategies
from cross_validation import TimeSeriesCV
from stationarity import StationarityTransformer
from exponential_smoothing import ExponentialSmoothingModels
from utils import TimeSeriesUtils

class TimeSeriesForecastingApp:
    """Веб-приложение для прогнозирования временных рядов"""
    
    def __init__(self):
        self.data = None
        self.utils = TimeSeriesUtils()
        
    def load_data(self):
        """Загрузка данных"""
        st.sidebar.header("📊 Загрузка данных")
        
        uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            try:
                self.data = pd.read_csv(uploaded_file)
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data.set_index('timestamp', inplace=True)
                elif 'date' in self.data.columns:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                    self.data.set_index('date', inplace=True)
                elif 'time' in self.data.columns:
                    self.data['time'] = pd.to_datetime(self.data['time'])
                    self.data.set_index('time', inplace=True)
                else:
                    # Если нет временной метки, создаем индексы
                    self.data.index = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
                
                st.sidebar.success(f"Данные загружены: {self.data.shape}")
                return True
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки данных: {e}")
                return False
        return False
    
    def sidebar_controls(self):
        """Элементы управления в сайдбаре"""
        st.sidebar.header("⚙️ Настройки")
        
        # Выбор целевой переменной
        if self.data is not None:
            target_options = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if target_options:
                target_column = st.sidebar.selectbox("Целевая переменная", target_options, index=0)
                
                # Горизонт прогноза
                horizon = st.sidebar.slider("Горизонт прогноза (дни)", 7, 90, 30)
                
                # Выбор типа декомпозиции
                decomposition_type = st.sidebar.radio("Тип декомпозиции", ['additive', 'multiplicative'])
                
                # Настройки преобразования
                st.sidebar.subheader("Преобразования")
                use_boxcox = st.sidebar.checkbox("Применить Бокса-Кокса")
                lambda_value = None
                if use_boxcox:
                    lambda_value = st.sidebar.number_input("λ для Бокса-Кокса", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
                
                return target_column, horizon, decomposition_type, use_boxcox, lambda_value
        
        return None, 30, 'additive', False, None
    
    def show_decomposition(self, target_column: str, decomposition_type: str):
        """Визуализация декомпозиции"""
        st.header("🔍 Декомпозиция временного ряда")
        
        if self.data is not None and target_column in self.data.columns:
            series = self.data[target_column].dropna()
            
            # Проверка на достаточную длину ряда
            if len(series) < 14:
                st.warning("⚠️ Слишком короткий ряд для декомпозиции. Нужно минимум 14 точек.")
                return
            
            # Анализ декомпозиции
            analyzer = DecompositionAnalyzer(series)
            
            col1, col2 = st.columns(2)
            
            with col1:
                period = st.selectbox("Период сезонности", [7, 30, 365], index=0, key="decomp_period_select")
                # Проверка на достаточную длину для выбранного периода
                if len(series) < 2 * period:
                    st.warning(f"⚠️ Для периода {period} нужно минимум {2 * period} точек данных.")
                    return
            
            with col2:
                if st.button("Выполнить декомпозицию", key="decomp_execute_button"):
                    with st.spinner("Выполняется декомпозиция..."):
                        try:
                            decomposition = analyzer.decompose(period=period, model=decomposition_type)
                            
                            if decomposition is not None:
                                # Визуализация
                                fig = self.utils.plot_decomposition(
                                    decomposition, 
                                    f"Декомпозиция ({decomposition_type})"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Анализ остатков
                                try:
                                    residual_analysis = analyzer.analyze_residuals(decomposition, decomposition_type)
                                    
                                    st.subheader("📊 Анализ остатков")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Среднее остатков", f"{residual_analysis['residuals_mean']:.4f}")
                                        st.metric("ADF p-value", f"{residual_analysis['stationarity']['ADF']['pvalue']:.4f}")
                                    
                                    with col2:
                                        st.metric("Стд. отклонение", f"{residual_analysis['residuals_std']:.4f}")
                                        st.metric("KPSS p-value", f"{residual_analysis['stationarity']['KPSS']['pvalue']:.4f}")
                                    
                                    with col3:
                                        st.metric("Нормальность (p-value)", f"{residual_analysis['normality']['pvalue']:.4f}")
                                        st.metric("Автокорреляция (p-value)", f"{residual_analysis['autocorrelation']['pvalue']:.4f}")
                                
                                except Exception as e:
                                    st.error(f"Ошибка анализа остатков: {e}")
                            
                        except Exception as e:
                            st.error(f"Ошибка декомпозиции: {e}")
                            st.info("""
                            **Возможные причины ошибки:**
                            - Слишком короткий временной ряд
                            - Недостаточно данных для выбранного периода сезонности
                            - Пропуски в данных
                            - Все значения одинаковые
                            """)
    
    def show_feature_engineering(self, target_column: str):
        """Feature Engineering"""
        st.header("🔧 Feature Engineering")
        
        if self.data is not None:
            feature_engineer = FeatureEngineer(self.data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lags = st.multiselect("Лаги", [1, 7, 14, 30], default=[1, 7, 30], key="lags_multiselect")
            
            with col2:
                windows = st.multiselect("Окна", [7, 30, 90], default=[7, 30], key="windows_multiselect")
            
            with col3:
                if st.button("Сгенерировать признаки", key="features_generate_button"):
                    with st.spinner("Генерация признаков..."):
                        try:
                            features_df = feature_engineer.create_all_features(target_column, lags, windows)
                            
                            st.success(f"Сгенерировано признаков: {len(features_df.columns)}")
                            
                            # Категории признаков
                            categories = feature_engineer.get_feature_categories()
                            
                            if categories:
                                for category, features in categories.items():
                                    if features:  # Показываем только непустые категории
                                        with st.expander(f"{category.capitalize()} признаки ({len(features)})"):
                                            st.write(features)
                            
                            # Информация о признаках
                            features_info = feature_engineer.get_features_info()
                            with st.expander("📊 Статистика признаков"):
                                st.write(f"Всего признаков: {features_info['total_features']}")
                                for category, count in features_info['categories_count'].items():
                                    st.write(f"- {category}: {count}")
                            
                            # Очистка признаков
                            with st.expander("🧹 Очистка признаков"):
                                clean_df = feature_engineer.clean_features()
                                st.write(f"После очистки: {len(clean_df.columns)} признаков")
                                st.write("Первые 5 строк очищенных данных:")
                                st.dataframe(clean_df.head())
                                
                        except Exception as e:
                            st.error(f"Ошибка при генерации признаков: {e}")
    
    def show_forecasting_strategies(self, target_column: str, horizon: int):
        """Сравнение стратегий прогнозирования"""
        st.header("🎯 Стратегии прогнозирования")
        
        if self.data is not None and target_column in self.data.columns:
            # Простая модель для демонстрации
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            model_choice = st.selectbox("Выбор модели", ['Linear Regression', 'Random Forest'], key="model_strategy_select")
            
            def model_factory():
                if model_choice == 'Linear Regression':
                    return LinearRegression()
                else:
                    return RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Разделение на train/test
            split_point = int(len(self.data) * 0.8)
            train_data = self.data.iloc[:split_point]
            test_data = self.data.iloc[split_point:split_point + horizon]
            
            # Базовые признаки
            basic_features = [target_column] + [f'lag_{i}' for i in [1, 7] if f'lag_{i}' in self.data.columns]
            
            if st.button("Сравнить стратегии", key="strategies_compare_button"):
                with st.spinner("Сравнение стратегий..."):
                    strategies = ForecastingStrategies(model_factory, target_column)
                    comparison = strategies.compare_strategies(train_data, test_data, horizon, basic_features)
                    
                    st.subheader("Результаты сравнения")
                    st.dataframe(comparison[['Strategy', 'MAE', 'RMSE', 'MAPE', 'Execution_Time']])
                    
                    # Визуализация прогнозов
                    fig = go.Figure()
                    
                    # Фактические значения
                    fig.add_trace(go.Scatter(x=test_data.index, y=test_data[target_column],
                                           name='Фактические значения', line=dict(color='blue')))
                    
                    # Прогнозы каждой стратегии
                    for strategy_name, result in strategies.results.items():
                        fig.add_trace(go.Scatter(x=test_data.index[:len(result['predictions'])], 
                                               y=result['predictions'],
                                               name=f'{strategy_name} прогноз'))
                    
                    fig.update_layout(title="Сравнение стратегий прогнозирования",
                                    xaxis_title="Дата", yaxis_title=target_column)
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_exponential_smoothing(self, target_column: str, horizon: int):
        """Экспоненциальное сглаживание с расширенными возможностями"""
        st.header("📈 Экспоненциальное сглаживание")
        
        if self.data is not None and target_column in self.data.columns:
            series = self.data[target_column].dropna()
            
            # Создаем вкладки внутри раздела сглаживания
            tab1, tab2, tab3 = st.tabs([
                "🔄 Сравнение моделей", 
                "📊 Прогноз с интервалами", 
                "📤 Экспорт модели"
            ])
            
            with tab1:
                self._show_model_comparison(series, horizon)
            
            with tab2:
                self._show_forecast_with_intervals(series, horizon)
            
            with tab3:
                self._show_model_export(series, horizon)

    def _show_model_comparison(self, series: pd.Series, horizon: int):
        """Сравнение моделей сглаживания"""
        st.subheader("Сравнение моделей сглаживания")
        
        if st.button("Сравнить модели сглаживания", key="smoothing_compare_main_button"):
            with st.spinner("Обучение моделей..."):
                es_models = ExponentialSmoothingModels(series)
                comparison = es_models.compare_models(forecast_horizon=horizon)
                
                st.dataframe(comparison)
                
                # Визуализация прогнозов
                fig = go.Figure()
                
                # Исторические данные
                fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                       name='Исторические данные', line=dict(color='gray')))
                
                # Прогнозы каждой модели
                for model_name, model_result in es_models.models.items():
                    if 'forecast' in model_result:
                        forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), 
                                                     periods=horizon, freq='D')
                        fig.add_trace(go.Scatter(x=forecast_dates, y=model_result['forecast'],
                                               name=f'{model_name} прогноз'))
                
                fig.update_layout(title="Прогнозы моделей экспоненциального сглаживания",
                                xaxis_title="Дата", yaxis_title="Значение")
                st.plotly_chart(fig, use_container_width=True)

    def _show_forecast_with_intervals(self, series: pd.Series, horizon: int):
        """Прогноз с доверительными интервалами"""
        st.subheader("Прогноз с доверительными интервалами")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Модель для прогноза",
                ['SES', 'Holt_Additive', 'Holt_Multiplicative', 'Naive'],
                index=1,
                key="forecast_model_main_select"
            )
        
        with col2:
            confidence_level = st.slider(
                "Уровень доверия",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="confidence_main_slider"
            )
        
        with col3:
            if st.button("Построить прогноз", key="forecast_main_button"):
                with st.spinner("Строим прогноз с доверительными интервалами..."):
                    es_models = ExponentialSmoothingModels(series)
                    
                    # Получаем прогноз с доверительными интервалами
                    result = es_models.forecast_with_confidence(
                        model_type=model_type,
                        forecast_horizon=horizon,
                        confidence_level=confidence_level
                    )
                    
                    if result:
                        # Визуализация
                        fig = self._plot_forecast_with_intervals(series, result, model_type, confidence_level)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Экспорт данных
                        forecast_df = es_models.export_to_dataframe(model_type, horizon)
                        
                        if not forecast_df.empty:
                            st.subheader("📈 Данные прогноза")
                            st.dataframe(forecast_df.style.format("{:.4f}"))
                            
                            # Скачивание данных
                            csv = forecast_df.reset_index().to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать прогноз (CSV)",
                                data=csv,
                                file_name=f"forecast_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                key="download_forecast_main_csv"
                            )
                    else:
                        st.error("Не удалось построить прогноз с выбранными параметрами")

    def _plot_forecast_with_intervals(self, series: pd.Series, result: dict, 
                                    model_type: str, confidence_level: float) -> go.Figure:
        """Визуализация прогноза с доверительными интервалами"""
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name='Исторические данные',
            line=dict(color='blue', width=2)
        ))
        
        # Прогноз
        forecast_dates = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=len(result['forecast']),
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=result['forecast'],
            name='Прогноз',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Доверительные интервалы
        if 'confidence_intervals' in result:
            ci = result['confidence_intervals']
            fig.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                y=ci['upper'].tolist() + ci['lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'Доверительный интервал {confidence_level*100:.0f}%',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"Прогноз с доверительными интервалами ({model_type})",
            xaxis_title="Дата",
            yaxis_title="Значение",
            hovermode='x unified',
            height=500
        )
        
        return fig

    def _show_model_export(self, series: pd.Series, horizon: int):
        """Экспорт параметров модели"""
        st.subheader("Экспорт модели и прогноза")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_model_type = st.selectbox(
                "Модель для экспорта",
                ['SES', 'Holt_Additive', 'Holt_Multiplicative'],
                index=1,
                key="export_model_main_select"
            )
        
        with col2:
            include_parameters = st.checkbox("Включить параметры модели", value=True, key="include_params_main_check")
        
        if st.button("Экспортировать результаты", key="export_main_button"):
            with st.spinner("Подготавливаем данные для экспорта..."):
                es_models = ExponentialSmoothingModels(series)
                
                # Полный экспорт
                export_data = es_models.export_forecast_results(
                    model_type=export_model_type,
                    forecast_horizon=horizon,
                    include_parameters=include_parameters
                )
                
                if export_data:
                    # Отображение параметров модели
                    if include_parameters and 'model_parameters' in export_data:
                        st.subheader("⚙️ Параметры модели")
                        
                        params = export_data['model_parameters']
                        param_cols = st.columns(3)
                        
                        param_items = list(params.items())
                        for i, (key, value) in enumerate(param_items):
                            with param_cols[i % 3]:
                                if isinstance(value, (int, float)):
                                    st.metric(key, f"{value:.4f}")
                                else:
                                    st.metric(key, str(value))
                    
                    # Статистики модели
                    if 'model_statistics' in export_data:
                        st.subheader("📊 Статистики модели")
                        stats = export_data['model_statistics']
                        
                        stat_cols = st.columns(4)
                        metrics = [
                            ('AIC', f"{stats.get('aic', 'N/A'):.2f}" if stats.get('aic') else 'N/A'),
                            ('BIC', f"{stats.get('bic', 'N/A'):.2f}" if stats.get('bic') else 'N/A'),
                            ('MAE остатков', f"{stats.get('residuals_mae', 0):.4f}"),
                            ('Стд. остатков', f"{stats.get('residuals_std', 0):.4f}")
                        ]
                        
                        for i, (name, value) in enumerate(metrics):
                            with stat_cols[i]:
                                st.metric(name, value)
                    
                    # Экспорт в JSON
                    def datetime_serializer(obj):
                        if isinstance(obj, (datetime, pd.Timestamp)):
                            return obj.isoformat()
                        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                    
                    json_data = json.dumps(export_data, default=datetime_serializer, indent=2, ensure_ascii=False)
                    
                    st.subheader("📄 JSON экспорт")
                    st.download_button(
                        label="📥 Скачать полный отчет (JSON)",
                        data=json_data,
                        file_name=f"model_export_{export_model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        key="download_json_main_button"
                    )
                    
                    # Предпросмотр JSON
                    with st.expander("Предпросмотр JSON данных"):
                        st.code(json_data, language='json')
                else:
                    st.error("Не удалось экспортировать данные модели")
    
    def run(self):
        """Запуск приложения"""
        st.set_page_config(page_title="Time Series Forecasting", page_icon="📈", layout="wide")
        
        st.title("📈 Прогнозирование временных рядов")
        st.markdown("---")
        
        # Загрузка данных
        if not self.load_data():
            st.info("👈 Загрузите CSV файл с временным рядом")
            return
        
        # Элементы управления
        target_column, horizon, decomposition_type, use_boxcox, lambda_value = self.sidebar_controls()
        
        if target_column is None:
            st.error("❌ В данных нет числовых столбцов для анализа")
            return
        
        # Упрощенная структура вкладок
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Декомпозиция", 
            "🔧 Признаки", 
            "🎯 Стратегии прогнозирования",
            "📈 Экспоненциальное сглаживание",
            "📋 Отчет"
        ])
        
        with tab1:
            self.show_decomposition(target_column, decomposition_type)
        
        with tab2:
            self.show_feature_engineering(target_column)
        
        with tab3:
            self.show_forecasting_strategies(target_column, horizon)
        
        with tab4:
            self.show_exponential_smoothing(target_column, horizon)
        
        with tab5:
            self.show_summary_report(target_column, horizon)
    
    def show_summary_report(self, target_column: str, horizon: int):
        """Сводный отчет"""
        st.header("📋 Сводный отчет")
        
        if self.data is not None and target_column in self.data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Статистики данных")
                st.write(self.data[target_column].describe())
            
            with col2:
                st.subheader("Информация о ряде")
                st.write(f"Период: {self.data.index[0]} - {self.data.index[-1]}")
                st.write(f"Длина ряда: {len(self.data)}")
                st.write(f"Пропуски: {self.data[target_column].isna().sum()}")
            
            # График ряда
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[target_column],
                                   name=target_column, line=dict(color='blue')))
            fig.update_layout(title=f"Временной ряд: {target_column}",
                            xaxis_title="Дата", yaxis_title=target_column)
            st.plotly_chart(fig, use_container_width=True)

# Запуск приложения
if __name__ == "__main__":
    app = TimeSeriesForecastingApp()
    app.run()