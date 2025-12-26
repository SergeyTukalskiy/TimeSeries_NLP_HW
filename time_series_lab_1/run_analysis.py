#!/usr/bin/env python3
"""
Основной скрипт для запуска полного анализа временных рядов
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.analyzer import TimeSeriesAnalyzer
from src.visualizer import TimeSeriesVisualizer
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("🚀 Запуск анализа временных рядов...")

    # 1. Загрузка данных
    print("📥 Этап 1: Загрузка данных...")
    loader = DataLoader()
    raw_data = loader.load_from_yahoo()
    print(f"Загружено данных: {raw_data.shape}")
    print(f"Колонки: {list(raw_data.columns)}")

    # 2. Очистка данных
    print("🧹 Этап 2: Очистка данных...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(raw_data)
    print(f"После очистки: {cleaned_data.shape}")

    # 3. Анализ
    print("📊 Этап 3: Анализ временных рядов...")
    analyzer = TimeSeriesAnalyzer(cleaned_data)

    # Базовая статистика
    stats = analyzer.get_descriptive_stats()
    print("\n📈 Базовая статистика:")
    print(stats)

    # Проверка стационарности
    print("\n📉 Тесты стационарности:")
    for column in cleaned_data.columns[:3]:  # Проверяем первые 3 чтобы не перегружать вывод
        print(f"\n{column}:")
        analyzer.test_stationarity(column)

    # Корреляционный анализ
    print("\n🔗 Корреляционный анализ:")
    correlations = analyzer.get_correlations()
    print(correlations.round(3))

    # 4. Визуализация
    print("\n🎨 Этап 4: Визуализация...")
    visualizer = TimeSeriesVisualizer(cleaned_data)

    # Создаем все графики
    visualizer.create_comprehensive_plots()

    # 5. Декомпозиция (только для первых 2 рядов чтобы не перегружать)
    print("\n🧩 Этап 5: Декомпозиция ряда...")
    for column in cleaned_data.columns[:2]:
        print(f"\nДекомпозиция {column}:")
        analyzer.decompose_time_series(column, period=30)

    # 6. Анализ автокорреляции (только для первых 2 рядов)
    print("\n🔍 Этап 6: Анализ автокорреляции...")
    for column in cleaned_data.columns[:2]:
        print(f"\nACF/PACF для {column}:")
        analyzer.plot_acf_pacf(column)

    # 7. Создание лаговых признаков и скользящих статистик
    print("\n⏳ Этап 7: Создание дополнительных признаков...")
    for column in cleaned_data.columns[:2]:
        df_lagged = analyzer.create_lagged_features(column, lags=[1, 7, 30])
        df_rolling = analyzer.calculate_rolling_stats(column, windows=[7, 30])
        print(f"Созданы лаговые признаки и скользящие статистики для {column}")

    print("\n✅ Анализ завершен! Результаты сохранены в папках 'output/' и 'data/'")
    print("\n📁 Созданные файлы:")
    print("   - data/raw_dataset.csv (исходные данные)")
    print("   - data/cleaned_dataset.csv (очищенные данные)")
    print("   - output/time_series_plots.png (графики временных рядов)")
    print("   - output/distributions.png (распределения)")
    print("   - output/correlation_heatmap.png (корреляции)")
    print("   - output/rolling_statistics.png (скользящие статистики)")
    print("   - output/boxplots.png (боксплоты)")
    print("   - output/interactive_plot.html (интерактивный график)")
    print("   - output/decomposition_*.png (декомпозиции)")
    print("   - output/acf_pacf_*.png (автокорреляционные функции)")


if __name__ == "__main__":
    main()