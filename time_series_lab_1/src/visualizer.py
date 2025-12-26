import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class TimeSeriesVisualizer:
    def __init__(self, df):
        self.df = df
        self.set_style()
        os.makedirs('output', exist_ok=True)

    def set_style(self):
        """Установка стиля графиков"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Настройки для matplotlib
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12

    def create_comprehensive_plots(self):
        """Создание всех основных графиков"""
        self.plot_time_series()
        self.plot_distributions()
        self.plot_correlation_heatmap()
        self.plot_rolling_statistics()
        self.plot_boxplots()
        self.create_interactive_plot()

    def plot_time_series(self):
        """Графики временных рядов - автоматическая подстройка под количество признаков"""
        n_cols = len(self.df.columns)

        # Вычисляем оптимальное количество строк и столбцов
        n_rows = (n_cols + 1) // 2  # Округление вверх
        n_cols_plot = 2 if n_cols > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(15, 5 * n_rows))

        # Если только один subplot, делаем axes итерируемым
        if n_cols == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols_plot > 1:
            axes = axes.ravel()
        elif n_rows == 1 and n_cols_plot > 1:
            axes = axes
        else:
            axes = [axes]

        for i, column in enumerate(self.df.columns):
            if n_rows > 1 or n_cols > 1:
                ax = axes[i]
            else:
                ax = axes

            self.df[column].plot(ax=ax, title=column, linewidth=1)
            ax.set_ylabel('Значение')
            ax.grid(True, alpha=0.3)

            # Добавляем скользящее среднее
            rolling_mean = self.df[column].rolling(window=30).mean()
            rolling_mean.plot(ax=ax, label='30-дневное среднее', alpha=0.8, linewidth=2)
            ax.legend()

        # Скрываем пустые subplots
        if n_rows > 1 or n_cols > 1:
            for j in range(i + 1, len(axes)):
                if j < len(axes):
                    axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('output/time_series_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Графики временных рядов сохранены")

    def plot_distributions(self):
        """Гистограммы распределений - автоматическая подстройка"""
        n_cols = len(self.df.columns)
        n_rows = (n_cols + 1) // 2
        n_cols_plot = 2 if n_cols > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(15, 5 * n_rows))

        # Обработка разных случаев размещения subplots
        if n_cols == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols_plot > 1:
            axes = axes.ravel()
        elif n_rows == 1 and n_cols_plot > 1:
            axes = axes
        else:
            axes = [axes]

        for i, column in enumerate(self.df.columns):
            if n_rows > 1 or n_cols > 1:
                ax = axes[i]
            else:
                ax = axes

            self.df[column].hist(bins=50, ax=ax, alpha=0.7)
            ax.set_title(f'Распределение {column}')
            ax.set_xlabel('Значение')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)

            # Добавляем линию плотности
            self.df[column].plot.density(ax=ax, secondary_y=True, alpha=0.7)

        # Скрываем пустые subplots
        if n_rows > 1 or n_cols > 1:
            for j in range(i + 1, len(axes)):
                if j < len(axes):
                    axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('output/distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Графики распределений сохранены")

    def plot_correlation_heatmap(self):
        """Тепловая карта корреляций"""
        corr_matrix = self.df.corr(method='spearman')

        # Основная heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, ax=ax, fmt='.3f')
        ax.set_title('Матрица корреляций (Spearman)')
        plt.tight_layout()
        plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Clustermap отдельно
        plt.figure(figsize=(12, 10))
        sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.3f')
        plt.title('Кластеризованная матрица корреляций')
        plt.tight_layout()
        plt.savefig('output/correlation_clustermap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Корреляционные матрицы сохранены")

    def plot_rolling_statistics(self, window=30):
        """Скользящие статистики - автоматическая подстройка"""
        n_cols = len(self.df.columns)
        n_rows = (n_cols + 1) // 2
        n_cols_plot = 2 if n_cols > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(15, 5 * n_rows))

        # Обработка разных случаев размещения subplots
        if n_cols == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols_plot > 1:
            axes = axes.ravel()
        elif n_rows == 1 and n_cols_plot > 1:
            axes = axes
        else:
            axes = [axes]

        for i, column in enumerate(self.df.columns):
            if n_rows > 1 or n_cols > 1:
                ax = axes[i]
            else:
                ax = axes

            # Исходный ряд и скользящее среднее
            self.df[column].plot(ax=ax, alpha=0.5, label='Исходный', linewidth=1)
            rolling_mean = self.df[column].rolling(window=window).mean()
            rolling_std = self.df[column].rolling(window=window).std()

            rolling_mean.plot(ax=ax, label=f'Скользящее среднее ({window}д)', linewidth=2)
            ax.fill_between(rolling_mean.index,
                            rolling_mean - rolling_std,
                            rolling_mean + rolling_std,
                            alpha=0.2, label=f'±1 STD')

            ax.set_title(f'Скользящие статистики: {column}')
            ax.set_ylabel('Значение')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Скрываем пустые subplots
        if n_rows > 1 or n_cols > 1:
            for j in range(i + 1, len(axes)):
                if j < len(axes):
                    axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('output/rolling_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Графики скользящих статистик сохранены")

    def plot_boxplots(self):
        """Boxplot для выявления выбросов - автоматическая подстройка"""
        n_cols = len(self.df.columns)
        n_rows = (n_cols + 1) // 2
        n_cols_plot = 2 if n_cols > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(15, 5 * n_rows))

        # Обработка разных случаев размещения subplots
        if n_cols == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols_plot > 1:
            axes = axes.ravel()
        elif n_rows == 1 and n_cols_plot > 1:
            axes = axes
        else:
            axes = [axes]

        for i, column in enumerate(self.df.columns):
            if n_rows > 1 or n_cols > 1:
                ax = axes[i]
            else:
                ax = axes

            self.df[[column]].boxplot(ax=ax)
            ax.set_title(f'Boxplot: {column}')
            ax.grid(True, alpha=0.3)

        # Скрываем пустые subplots
        if n_rows > 1 or n_cols > 1:
            for j in range(i + 1, len(axes)):
                if j < len(axes):
                    axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('output/boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Boxplots сохранены")

    def create_interactive_plot(self):
        """Интерактивный график Plotly"""
        fig = go.Figure()

        for column in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df.index,
                y=self.df[column],
                name=column,
                mode='lines'
            ))

        fig.update_layout(
            title='Интерактивный график временных рядов',
            xaxis_title='Дата',
            yaxis_title='Значение',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )

        fig.write_html('output/interactive_plot.html')
        print("✅ Интерактивный график сохранен")

    def plot_individual_time_series(self, column):
        """График для отдельного временного ряда"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df[column].plot(ax=ax, title=f'Временной ряд: {column}', linewidth=2)
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3)

        # Добавляем скользящее среднее
        rolling_mean = self.df[column].rolling(window=30).mean()
        rolling_mean.plot(ax=ax, label='30-дневное среднее', alpha=0.8, linewidth=2)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'output/{column}_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ График {column} сохранен")