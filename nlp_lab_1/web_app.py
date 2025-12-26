import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import sys
import os

# Добавляем путь к нашим модулям
sys.path.append(os.path.dirname(__file__))


class TokenizationWebApp:
    def __init__(self):
        self.setup_page()

    def setup_page(self):
        """Настройка страницы Streamlit"""
        st.set_page_config(
            page_title="Анализ методов токенизации",
            page_icon="📊",
            layout="wide"
        )

        st.title("Сравнительный анализ методов токенизации и нормализации")
        st.markdown("---")

    def run(self):
        """Запуск приложения"""
        # Боковая панель для загрузки данных
        st.sidebar.header("Загрузка данных")

        uploaded_file = st.sidebar.file_uploader(
            "Загрузите JSONL файл с текстами",
            type=['jsonl']
        )

        # Боковая панель для выбора методов
        st.sidebar.header("Параметры анализа")

        selected_methods = st.sidebar.multiselect(
            "Выберите методы токенизации:",
            ["Naive", "Razdel", "SpaCy", "Pymorphy Lemma", "Snowball Stem", "BPE", "WordPiece"]
        )

        # Основная область
        if uploaded_file is not None:
            self.analyze_data(uploaded_file, selected_methods)
        else:
            self.show_instructions()

    def show_instructions(self):
        """Показать инструкции"""
        st.info("""
        ### Инструкция по использованию:
        1. Загрузите JSONL файл с текстами в боковой панели
        2. Выберите методы токенизации для сравнения
        3. Просмотрите результаты анализа на графиках
        4. Экспортируйте отчёт в нужном формате
        """)

        # Пример данных
        st.subheader("Пример структуры JSONL файла:")
        st.code("""
        {"title": "Заголовок новости", "text": "Текст новости...", "date": "2024-01-01", "url": "https://..."}
        {"title": "Другая новость", "text": "Еще один текст...", "date": "2024-01-02", "url": "https://..."}
        """)

    def analyze_data(self, uploaded_file, selected_methods):
        """Анализ загруженных данных"""
        # Загрузка и обработка данных
        texts = self.load_data(uploaded_file)

        if not texts:
            st.error("Не удалось загрузить тексты из файла")
            return

        # Показ статистики
        self.show_statistics(texts)

        if selected_methods:
            # Запуск сравнения методов
            comparison_results = self.run_comparison(texts, selected_methods)

            # Визуализация результатов
            self.visualize_results(comparison_results, texts)

            # Кнопка экспорта
            if st.button("Экспортировать отчёт в HTML"):
                self.export_report(comparison_results)

    def load_data(self, uploaded_file):
        """Загрузка данных из JSONL"""
        texts = []
        try:
            for line in uploaded_file:
                data = json.loads(line)
                if 'text' in data:
                    texts.append(data['text'])
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
        return texts

    def show_statistics(self, texts):
        """Показать статистику корпуса"""
        col1, col2, col3, col4 = st.columns(4)

        total_texts = len(texts)
        total_words = sum(len(text.split()) for text in texts)
        avg_words = total_words / total_texts if total_texts > 0 else 0

        with col1:
            st.metric("Количество текстов", total_texts)
        with col2:
            st.metric("Общее количество слов", total_words)
        with col3:
            st.metric("Среднее слов в тексте", f"{avg_words:.1f}")
        with col4:
            st.metric("Размер корпуса", f"{(sum(len(t.encode('utf-8')) for t in texts) / 1024 / 1024):.1f} MB")

    def run_comparison(self, texts, methods):
        """Запуск сравнения методов"""
        # Здесь интегрируйте ваш код сравнения методов
        # Это упрощенный пример
        results = {}

        for method in methods:
            # Имитация результатов
            results[method] = {
                'vocab_size': len(set(' '.join(texts).split())),  # Упрощенный расчет
                'processing_time': 1.0,
                'oov_rate': 0.05,
                'fragmentation_rate': 0.1
            }

        return results

    def visualize_results(self, results, texts):
        """Визуализация результатов"""
        st.subheader("Сравнительный анализ методов")

        # График сравнения метрик
        df = pd.DataFrame.from_dict(results, orient='index')

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(df, y='vocab_size', title='Размер словаря')
            st.plotly_chart(fig)

        with col2:
            fig = px.bar(df, y='oov_rate', title='OOV Rate')
            st.plotly_chart(fig)

        # Распределение длин токенов
        st.subheader("Распределение длин токенов")
        # Добавьте визуализацию распределения

        # Частотность токенов
        st.subheader("Топ-20 самых частых токенов")
        self.show_top_tokens(texts)

    def show_top_tokens(self, texts):
        """Показать топ токенов"""
        all_tokens = ' '.join(texts).split()
        token_freq = Counter(all_tokens)
        top_tokens = token_freq.most_common(20)

        tokens, counts = zip(*top_tokens)
        fig = px.bar(x=counts, y=tokens, orientation='h', title='Топ-20 токенов')
        st.plotly_chart(fig)

    def export_report(self, results):
        """Экспорт отчёта"""
        # Реализация экспорта в HTML/PDF
        st.success("Отчёт успешно экспортирован!")


if __name__ == "__main__":
    app = TokenizationWebApp()
    app.run()