import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.analyzer import TimeSeriesAnalyzer
from src.visualizer import TimeSeriesVisualizer

# Настройки страницы
st.set_page_config(
    page_title="Анализатор временных рядов",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Глобальная переменная для отслеживания уникальных ключей
if 'plot_key_counter' not in st.session_state:
    st.session_state.plot_key_counter = 0


def get_unique_key(prefix="plot"):
    """Генерирует уникальный ключ для элементов Streamlit"""
    st.session_state.plot_key_counter += 1
    return f"{prefix}_{st.session_state.plot_key_counter}"


def safe_correlation_matrix(df, columns):
    """Безопасное создание матрицы корреляций без дубликатов"""
    # Убираем дубликаты
    unique_columns = list(dict.fromkeys(columns))
    if len(unique_columns) < 2:
        return None
    return df[unique_columns].corr()


def main():
    st.title("🔍 Интерактивный анализ временных рядов")
    st.markdown("Лабораторная работа по анализу временных рядов")

    # Сайдбар с настройками
    st.sidebar.header("⚙️ Настройки анализа")

    # Загрузка данных
    st.sidebar.subheader("📥 Загрузка данных")
    data_source = st.sidebar.radio("Источник данных:",
                                   ["Пример данных (Yahoo Finance)", "Загрузить CSV файл"],
                                   key="data_source")

    if data_source == "Пример данных (Yahoo Finance)":
        if st.sidebar.button("Загрузить пример данных", key="load_example"):
            with st.spinner("Загрузка данных с Yahoo Finance..."):
                try:
                    loader = DataLoader()
                    df = loader.load_from_yahoo()
                    cleaner = DataCleaner()
                    df_clean = cleaner.clean_data(df)
                    st.session_state.df = df_clean
                    st.session_state.data_loaded = True
                    st.session_state.available_columns = df_clean.columns.tolist()
                    st.sidebar.success("Данные успешно загружены!")
                except Exception as e:
                    st.sidebar.error(f"Ошибка загрузки: {e}")
    else:
        uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=['csv'], key="file_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                cleaner = DataCleaner()
                df_clean = cleaner.clean_data(df)
                st.session_state.df = df_clean
                st.session_state.data_loaded = True
                st.session_state.available_columns = df_clean.columns.tolist()
                st.sidebar.success("Данные успешно загружены!")
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки файла: {e}")

    # Если данные загружены, показываем анализ
    if 'df' in st.session_state and st.session_state.get('data_loaded', False):
        df = st.session_state.df
        available_columns = st.session_state.available_columns

        # Выбор переменных
        st.sidebar.subheader("📊 Выбор переменных")
        target_var = st.sidebar.selectbox("Целевая переменная:", available_columns, key="target_select")

        # Доступные признаки (исключая целевую переменную)
        available_features = [col for col in available_columns if col != target_var]
        default_features = available_features[:min(3, len(available_features))]

        feature_vars = st.sidebar.multiselect("Признаки:", available_features,
                                              default=default_features, key="feature_select")

        # Настройки анализа
        st.sidebar.subheader("🔧 Параметры анализа")
        decomposition_period = st.sidebar.number_input("Период сезонности:",
                                                       min_value=2, max_value=365, value=30,
                                                       key="decomp_period")
        max_lag = st.sidebar.slider("Макс. лаг для ACF/PACF:", 10, 100, 40, key="max_lag")
        rolling_window = st.sidebar.slider("Окно скользящего среднего:", 5, 100, 30, key="rolling_window")

        # Основная область контента
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Обзор данных", "📊 Визуализация",
                                                "🔍 Анализ", "📉 Стационарность", "📋 Отчет"])

        with tab1:
            st.header("Обзор данных")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Основная информация")
                st.write(f"**Размер данных:** {df.shape}")
                st.write(f"**Период:** {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")
                st.write(f"**Количество наблюдений:** {len(df)}")
                st.write(f"**Колонки:** {', '.join(df.columns)}")

                st.subheader("Пропущенные значения")
                missing_data = df.isnull().sum()
                st.write(missing_data)

            with col2:
                st.subheader("Статистика")
                st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            st.header("Визуализация данных")

            # График временных рядов
            st.subheader("Временные ряды")
            fig_ts = go.Figure()

            # Добавляем целевую переменную первой
            fig_ts.add_trace(go.Scatter(x=df.index, y=df[target_var],
                                        name=f"{target_var} (целевая)", mode='lines', line=dict(width=3)))

            # Добавляем признаки
            for column in feature_vars:
                fig_ts.add_trace(go.Scatter(x=df.index, y=df[column],
                                            name=column, mode='lines', opacity=0.7))

            fig_ts.update_layout(title="Временные ряды", height=500)
            st.plotly_chart(fig_ts, use_container_width=True, key=get_unique_key("timeseries"))

            # Распределения и Boxplots
            st.subheader("Распределения и выбросы")
            col1, col2 = st.columns(2)

            with col1:
                # Распределения для целевой переменной и первых 2 признаков
                plot_columns = [target_var] + feature_vars[:2]
                for i, column in enumerate(plot_columns):
                    if i >= 3:  # Ограничиваем количество графиков
                        break
                    fig_hist = px.histogram(df, x=column, title=f"Распределение {column}")
                    st.plotly_chart(fig_hist, use_container_width=True, key=get_unique_key(f"hist_{column}"))

            with col2:
                # Boxplot для целевой переменной и первых 2 признаков
                fig_box = go.Figure()
                plot_columns = [target_var] + feature_vars[:2]
                for i, column in enumerate(plot_columns):
                    if i >= 3:  # Ограничиваем количество графиков
                        break
                    fig_box.add_trace(go.Box(y=df[column], name=column))
                fig_box.update_layout(title="Boxplot - распределение и выбросы")
                st.plotly_chart(fig_box, use_container_width=True, key=get_unique_key("boxplot"))

            # Корреляционная матрица
            st.subheader("Корреляционная матрица")
            # Используем безопасную функцию для создания матрицы корреляций
            corr_matrix = safe_correlation_matrix(df, [target_var] + feature_vars)

            if corr_matrix is not None and len(corr_matrix) > 1:
                fig_corr = px.imshow(corr_matrix,
                                     text_auto=True,
                                     aspect="auto",
                                     title="Матрица корреляций",
                                     color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True, key=get_unique_key("correlation"))
            else:
                st.info("Для построения корреляционной матрицы нужно выбрать хотя бы 2 различные переменные")

        with tab3:
            st.header("Анализ временных рядов")

            # Декомпозиция
            st.subheader("Декомпозиция ряда")
            try:
                # Убедимся, что данных достаточно для декомпозиции
                if len(df[target_var].dropna()) > decomposition_period * 2:
                    decomposition = seasonal_decompose(df[target_var].dropna(),
                                                       period=decomposition_period,
                                                       extrapolate_trend='freq')

                    fig_dec = make_subplots(rows=4, cols=1,
                                            subplot_titles=['Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'])

                    fig_dec.add_trace(go.Scatter(x=df.index, y=df[target_var], name='Исходный'), row=1, col=1)
                    fig_dec.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд'),
                                      row=2, col=1)
                    fig_dec.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность'), row=3,
                        col=1)
                    fig_dec.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Остатки'),
                                      row=4, col=1)

                    fig_dec.update_layout(height=800, showlegend=False, title_text=f"Декомпозиция: {target_var}")
                    st.plotly_chart(fig_dec, use_container_width=True, key=get_unique_key("decomposition"))
                else:
                    st.warning(
                        f"Для декомпозиции с периодом {decomposition_period} нужно больше данных. Уменьшите период или выберите ряд с большей историей.")

            except Exception as e:
                st.error(f"Ошибка при декомпозиции: {e}")
                st.info("Попробуйте изменить период сезонности или выберите другую переменную")

            # ACF и PACF
            st.subheader("Автокорреляционные функции")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**ACF (Автокорреляционная функция) - {target_var}**")
                try:
                    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                    plot_acf(df[target_var].dropna(), lags=max_lag, ax=ax_acf)
                    ax_acf.set_title(f"ACF: {target_var}")
                    ax_acf.grid(True, alpha=0.3)
                    st.pyplot(fig_acf)  # Убрали параметр key
                    plt.close(fig_acf)  # Закрываем figure чтобы освободить память
                except Exception as e:
                    st.error(f"Ошибка построения ACF: {e}")

            with col2:
                st.write(f"**PACF (Частная автокорреляционная функция) - {target_var}**")
                try:
                    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                    plot_pacf(df[target_var].dropna(), lags=max_lag, ax=ax_pacf, method='ywm')
                    ax_pacf.set_title(f"PACF: {target_var}")
                    ax_pacf.grid(True, alpha=0.3)
                    st.pyplot(fig_pacf)  # Убрали параметр key
                    plt.close(fig_pacf)  # Закрываем figure чтобы освободить память
                except Exception as e:
                    st.error(f"Ошибка построения PACF: {e}")

        with tab4:
            st.header("Анализ стационарности")

            # Тесты стационарности
            st.subheader("Статистические тесты")

            # Ограничиваем количество анализируемых переменных
            analysis_vars = [target_var] + feature_vars[:3]  # Целевая + до 3 признаков

            for column in analysis_vars:
                st.write(f"**{column}**")

                try:
                    # Создаем временный анализатор для теста
                    temp_analyzer = TimeSeriesAnalyzer(df)
                    is_adf, is_kpss = temp_analyzer.test_stationarity(column)

                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"Переменная: {column}")
                    with col2:
                        status_color = "🟢" if is_adf else "🔴"
                        st.metric("ADF тест", "Стационарен" if is_adf else "Нестационарен",
                                  delta=status_color, delta_color="normal" if is_adf else "inverse")
                    with col3:
                        status_color = "🟢" if is_kpss else "🔴"
                        st.metric("KPSS тест", "Стационарен" if is_kpss else "Нестационарен",
                                  delta=status_color, delta_color="normal" if is_kpss else "inverse")

                except Exception as e:
                    st.error(f"Ошибка при анализе {column}: {e}")

                st.markdown("---")

            # Скользящая статистика
            st.subheader("Скользящая статистика")
            try:
                df_rolling = df[target_var].rolling(window=rolling_window)

                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=df.index, y=df[target_var],
                                              name='Исходный', line=dict(color='blue')))
                fig_roll.add_trace(go.Scatter(x=df.index, y=df_rolling.mean(),
                                              name=f'Скользящее среднее ({rolling_window})',
                                              line=dict(color='red')))
                fig_roll.add_trace(go.Scatter(x=df.index, y=df_rolling.std(),
                                              name=f'Скользящее STD ({rolling_window})',
                                              line=dict(color='green')))

                fig_roll.update_layout(title=f"Скользящая статистика: {target_var}", height=500)
                st.plotly_chart(fig_roll, use_container_width=True, key=get_unique_key("rolling"))
            except Exception as e:
                st.error(f"Ошибка построения скользящей статистики: {e}")

        with tab5:
            st.header("Итоговый отчет")

            if st.button("Сгенерировать полный отчет", key="report_button"):
                with st.spinner("Генерация отчета..."):
                    try:
                        # Создаем анализатор и визуализатор
                        analyzer = TimeSeriesAnalyzer(df)

                        # Собираем всю информацию
                        st.subheader("📋 Результаты анализа")

                        # Основная статистика
                        st.write("### Базовая статистика")
                        stats_df = analyzer.get_descriptive_stats()
                        st.dataframe(stats_df, use_container_width=True)

                        # Корреляции
                        st.write("### Корреляционный анализ")
                        corr_matrix = safe_correlation_matrix(df, [target_var] + feature_vars)

                        if corr_matrix is not None and len(corr_matrix) > 1:
                            fig_corr_report = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                                        title="Матрица корреляций", color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig_corr_report, use_container_width=True,
                                            key=get_unique_key("corr_report"))

                            # Анализ корреляций
                            st.write("**Сильные корреляции (>0.7):**")
                            strong_corrs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i + 1, len(corr_matrix.columns)):
                                    corr_val = corr_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.7:
                                        strong_corrs.append({
                                            'Переменная 1': corr_matrix.columns[i],
                                            'Переменная 2': corr_matrix.columns[j],
                                            'Корреляция': f"{corr_val:.3f}"
                                        })

                            if strong_corrs:
                                st.table(pd.DataFrame(strong_corrs))
                            else:
                                st.write("Нет сильных корреляций между переменными")
                        else:
                            st.info("Недостаточно переменных для корреляционного анализа")

                        # Стационарность
                        st.write("### Анализ стационарности")
                        stationarity_results = []
                        analysis_vars = [target_var] + feature_vars[:3]

                        for column in analysis_vars:
                            try:
                                is_adf, is_kpss = analyzer.test_stationarity(column)
                                stationarity_results.append({
                                    'Переменная': column,
                                    'ADF': 'Стационарен' if is_adf else 'Нестационарен',
                                    'KPSS': 'Стационарен' if is_kpss else 'Нестационарен',
                                    'Рекомендация': 'Готов к моделированию' if (
                                                is_adf and is_kpss) else 'Требует дифференцирования'
                                })
                            except Exception as e:
                                stationarity_results.append({
                                    'Переменная': column,
                                    'ADF': f'Ошибка',
                                    'KPSS': f'Ошибка',
                                    'Рекомендация': f'Требует проверки: {str(e)[:50]}...'
                                })

                        st.table(pd.DataFrame(stationarity_results))

                        # Ключевые выводы
                        st.write("### Ключевые выводы")

                        # Анализ целевой переменной
                        st.write(f"**Анализ целевой переменной ({target_var}):**")
                        try:
                            is_adf, is_kpss = analyzer.test_stationarity(target_var)
                            if is_adf and is_kpss:
                                st.success("✅ Ряд стационарен - можно использовать для моделирования")
                            else:
                                st.warning("⚠️ Ряд нестационарен - рекомендуется дифференцирование")
                        except Exception as e:
                            st.error(f"❌ Не удалось провести анализ стационарности: {e}")

                        # Рекомендации
                        st.write("**Рекомендации для дальнейшего анализа:**")
                        st.markdown("""
                        - Для нестационарных рядов применить дифференцирование
                        - Использовать значимые лаги из ACF/PACF для feature engineering
                        - Учесть сезонность при построении моделей
                        - Протестировать различные методы прогнозирования (ARIMA, Prophet, LSTM)
                        """)

                        st.success("✅ Отчет сгенерирован!")

                    except Exception as e:
                        st.error(f"❌ Ошибка при генерации отчета: {e}")

    else:
        st.info("👆 Пожалуйста, загрузите данные для начала анализа")
        st.markdown("""
        ### Инструкция:
        1. Выберите источник данных в боковой панели
        2. Загрузите CSV файл или используйте пример данных
        3. Настройте параметры анализа
        4. Исследуйте данные через вкладки

        ### Пример данных включают:
        - Цена на нефть Brent
        - Курс USD/RUB
        - Индекс МосБиржи (MOEX)
        - Цена на золото
        - Акции Сбербанка
        - Акции Газпрома
        """)


if __name__ == "__main__":
    main()