import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os
import networkx as nx
from sklearn.cluster import KMeans

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.semantic_analysis import SemanticAnalyzer
from src.distributed_models import DistributedModels
from src.dimensionality_reduction import DimensionalityReducer

class VectorSpaceExplorer:
    """Веб-интерфейс для анализа векторных пространств"""
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.dimensionality_reducer = DimensionalityReducer()
        self.models = {}
        
    def load_models(self, models_dict: Dict[str, Any]):
        """Загрузка обученных моделей"""
        self.models = models_dict
        
    def render_sidebar(self):
        """Отрисовка боковой панели"""
        st.sidebar.title("🔍 Анализатор векторных пространств")
        
        # Выбор модели
        if self.models:
            model_names = list(self.models.keys())
            selected_model_name = st.sidebar.selectbox("Выберите модель:", model_names)
            self.selected_model = self.models[selected_model_name]['model']
        else:
            st.sidebar.warning("Модели не загружены")
            self.selected_model = None
            
        # Навигация
        page = st.sidebar.radio(
            "Разделы:",
            ["Векторная арифметика", "Семантическое сходство", "Семантические оси", 
             "Визуализация", "Динамический отчет"]
        )
        
        return page
    
    def render_vector_arithmetic(self):
        """Интерфейс векторной арифметики с отдельными полями для положительных и отрицательных слов"""
        st.header("🧮 Интерактивная векторная арифметика")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_words = st.text_area(
                "Положительные слова (через запятую):", 
                "король, женщина",
                help="Слова, которые будут прибавлены к вектору"
            )
        with col2:
            negative_words = st.text_area(
                "Отрицательные слова (через запятую):", 
                "мужчина",
                help="Слова, которые будут вычтены из вектора"
            )
        with col3:
            topn = st.slider("Количество результатов:", 1, 20, 10)
        
        if st.button("Вычислить"):
            if self.selected_model:
                positive = [w.strip() for w in positive_words.split(',') if w.strip()]
                negative = [w.strip() for w in negative_words.split(',') if w.strip()]
                
                try:
                    results = self.semantic_analyzer.vector_arithmetic(
                        self.selected_model, positive, negative, topn
                    )
                    
                    if results:
                        # Отображаем формулу
                        formula_parts = []
                        if positive:
                            formula_parts.append(" + ".join(positive))
                        if negative:
                            formula_parts.append(" - " + " - ".join(negative))
                        
                        formula = "".join(formula_parts)
                        st.subheader(f"Результат: {formula}")
                        
                        # Таблица результатов
                        df = pd.DataFrame(results, columns=['Слово', 'Сходство'])
                        st.dataframe(df)
                        
                        # Визуализация
                        fig = px.bar(df, x='Сходство', y='Слово', orientation='h',
                                    title=f"Результаты векторной арифметики: {formula}")
                        st.plotly_chart(fig)
                        
                        # Ближайшие соседи для каждого слова
                        st.subheader("🔍 Ближайшие соседи для входных слов")
                        
                        # Для положительных слов
                        if positive:
                            st.write("**Положительные слова:**")
                            pos_cols = st.columns(len(positive))
                            for i, word in enumerate(positive):
                                with pos_cols[i]:
                                    neighbors = self.semantic_analyzer.find_similar_words(
                                        self.selected_model, word, 5
                                    )
                                    if neighbors:
                                        st.write(f"Ближайшие к '{word}':")
                                        neighbor_df = pd.DataFrame(neighbors, columns=['Слово', 'Сходство'])
                                        st.dataframe(neighbor_df)
                        
                        # Для отрицательных слов
                        if negative:
                            st.write("**Отрицательные слова:**")
                            neg_cols = st.columns(len(negative))
                            for i, word in enumerate(negative):
                                with neg_cols[i]:
                                    neighbors = self.semantic_analyzer.find_similar_words(
                                        self.selected_model, word, 5
                                    )
                                    if neighbors:
                                        st.write(f"Ближайшие к '{word}':")
                                        neighbor_df = pd.DataFrame(neighbors, columns=['Слово', 'Сходство'])
                                        st.dataframe(neighbor_df)
                        
                    else:
                        st.error("Не удалось выполнить операцию. Проверьте, что слова есть в словаре модели.")
                except Exception as e:
                    st.error(f"Ошибка при выполнении векторной арифметики: {e}")
            else:
                st.error("Модель не выбрана")
    
    def render_semantic_similarity(self):
        """Интерфейс семантического сходства с графами и анализом расстояний"""
        st.header("📊 Эксперименты с семантическим сходством")
        
        tab1, tab2, tab3 = st.tabs(["Калькулятор расстояний", "Граф семантических связей", "Анализ распределения расстояний"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                word1 = st.text_input("Первое слово:", "компьютер")
            with col2:
                word2 = st.text_input("Второе слово:", "ноутбук")
            
            if st.button("Вычислить сходство") and self.selected_model:
                try:
                    similarity = self.semantic_analyzer.calculate_word_distance(
                        self.selected_model, word1, word2
                    )
                    
                    st.metric("Косинусное сходство", f"{similarity:.4f}")
                    
                    # Ближайшие соседи для обоих слов
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        neighbors1 = self.semantic_analyzer.find_similar_words(
                            self.selected_model, word1, 10
                        )
                        st.subheader(f"Ближайшие к '{word1}':")
                        df1 = pd.DataFrame(neighbors1, columns=['Слово', 'Сходство'])
                        st.dataframe(df1)
                    
                    with col2:
                        neighbors2 = self.semantic_analyzer.find_similar_words(
                            self.selected_model, word2, 10
                        )
                        st.subheader(f"Ближайшие к '{word2}':")
                        df2 = pd.DataFrame(neighbors2, columns=['Слово', 'Сходство'])
                        st.dataframe(df2)
                        
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        
        with tab2:
            seed_words = st.text_area(
                "Начальные слова для графа (через запятую):",
                "компьютер, программа, данные, алгоритм, сеть, искусственный, интеллект"
            )
            threshold = st.slider("Порог сходства для связей:", 0.1, 0.9, 0.5)
            depth = st.slider("Глубина графа:", 1, 3, 2)
            
            if st.button("Построить граф") and self.selected_model:
                seed_list = [w.strip() for w in seed_words.split(',') if w.strip()]
                
                network = self.semantic_analyzer.create_semantic_network(
                    self.selected_model, seed_list, depth=depth, threshold=threshold
                )
                
                st.metric("Узлы графа", network['metrics']['nodes_count'])
                st.metric("Связи графа", network['metrics']['edges_count'])
                st.metric("Плотность графа", f"{network['metrics']['density']:.4f}")
                
                # Отображение таблицы связей
                links_df = pd.DataFrame(network['links'])
                if not links_df.empty:
                    st.subheader("Семантические связи (топ-20 по сходству):")
                    st.dataframe(links_df.nlargest(20, 'value'))
                else:
                    st.warning("Не найдено связей с заданным порогом сходства. Попробуйте уменьшить порог.")
        
        with tab3:
            st.info("Анализ распределения расстояний вычисляет косинусные сходства между случайными парами слов из словаря модели.")
            
            if st.button("Анализировать распределение расстояний") and self.selected_model:
                with st.spinner("Анализ распределения расстояний..."):
                    distribution = self.semantic_analyzer.analyze_distance_distribution(
                        self.selected_model
                    )
                    
                    st.subheader("📊 Статистика распределения расстояний")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Среднее сходство", f"{distribution['mean']:.4f}")
                    col2.metric("Стандартное отклонение", f"{distribution['std']:.4f}")
                    col3.metric("Минимальное", f"{distribution['min']:.4f}")
                    col4.metric("Максимальное", f"{distribution['max']:.4f}")
                    
                    st.write(f"**Анализировано пар:** {distribution['n_pairs']}")
                    st.write(f"**Размер выборки:** {distribution.get('sample_size', 'N/A')}")
                    
                    # Визуализация распределения
                    if distribution['distances']:
                        fig = px.histogram(
                            x=distribution['distances'], 
                            title="Распределение семантических расстояний",
                            labels={'x': 'Косинусное сходство', 'y': 'Частота'},
                            nbins=20
                        )
                        fig.add_vline(x=distribution['mean'], line_dash="dash", line_color="red", 
                                    annotation_text=f"Среднее: {distribution['mean']:.4f}")
                        st.plotly_chart(fig)
                    
                    # Интерпретация результатов
                    st.subheader("📈 Интерпретация результатов")
                    mean_similarity = distribution['mean']
                    
                    if mean_similarity > 0.3:
                        st.success("✅ Высокое среднее сходство: модель хорошо улавливает семантические связи между словами")
                    elif mean_similarity > 0.1:
                        st.info("ℹ️ Умеренное среднее сходство: модель различает семантику, но есть пространство для улучшения")
                    else:
                        st.warning("⚠️ Низкое среднее сходство: модель плохо улавливает семантические связи")
    
    def render_semantic_axes(self):
        """Исправленный интерфейс семантических осей"""
        st.header("📈 Визуализация семантических осей")
        
        st.info("""
        **Семантические оси** показывают распределение слов между двумя концептами.
        - **Отрицательные значения**: ближе к первому слову
        - **Положительные значения**: ближе ко второму слову
        - **Нулевые значения**: нейтральные или равноудаленные
        """)
        
        # ТЕСТИРУЕМ РАЗНЫЕ ОСИ
        axis_options = {
            "Гендерная": ("мужчина", "женщина"),
            "Время": ("прошлое", "будущее"),
            "Оценка": ("плохой", "хороший"),
            "Размер": ("маленький", "большой"),
        }
        
        selected_axis = st.selectbox("Выберите семантическую ось:", list(axis_options.keys()))
        
        word1, word2 = axis_options[selected_axis]
        st.write(f"**Ось:** {word1} ←→ {word2}")
        
        # Тестовые слова для разных осей
        test_words_config = {
            "Гендерная": "мужчина, женщина, парень, девушка, отец, мать, сын, дочь, брат, сестра, дядя, тетя",
            "Время": "вчера, сегодня, завтра, старый, новый, древний, современный, прошлый, будущий, настоящий",
            "Оценка": "ужасный, прекрасный, отличный, плохой, хороший, скверный, замечательный, худший, лучший",
            "Размер": "малый, крупный, огромный, крошечный, гигантский, миниатюрный, масштабный, компактный",
        }
        
        test_words = st.text_area(
            "Слова для анализа:",
            test_words_config.get(selected_axis, "слово1, слово2, слово3")
        )
        
        col1, col2 = st.columns(2)
        with col1:
            use_advanced = st.checkbox("Использовать улучшенный метод", value=True)
        with col2:
            show_debug = st.checkbox("Показать отладочную информацию")
        
        if st.button("Анализировать ось") and self.selected_model:
            # Проверяем наличие слов оси
            missing_axis_words = []
            for word in [word1, word2]:
                if word not in self.selected_model.wv.key_to_index:
                    missing_axis_words.append(word)
            
            if missing_axis_words:
                st.error(f"Слова оси не найдены в модели: {', '.join(missing_axis_words)}")
                available_words = list(self.selected_model.wv.key_to_index.keys())[:20]
                st.info("Доступные слова (первые 20): " + ", ".join(available_words))
                return
            
            axis_pairs = {selected_axis: (word1, word2)}
            words_to_test = [w.strip() for w in test_words.split(',') if w.strip()]
            
            # Проверяем тестовые слова
            available_test_words = [w for w in words_to_test if w in self.selected_model.wv.key_to_index]
            missing_test_words = [w for w in words_to_test if w not in self.selected_model.wv.key_to_index]
            
            if missing_test_words:
                st.warning(f"Некоторые тестовые слова не найдены: {', '.join(missing_test_words[:5])}")
            
            if len(available_test_words) < 3:
                st.error("Слишком мало тестовых слов найдено в модели")
                return
            
            with st.spinner("Анализ семантической оси..."):
                try:
                    if use_advanced:
                        results = self.semantic_analyzer.analyze_semantic_axes_advanced(
                            self.selected_model, axis_pairs, available_test_words
                        )
                    else:
                        results = self.semantic_analyzer.analyze_semantic_axes(
                            self.selected_model, axis_pairs, available_test_words
                        )
                    
                    if selected_axis in results:
                        axis_result = results[selected_axis]
                        projections = axis_result['projections']
                        
                        if not projections:
                            st.error("Не удалось вычислить проекции")
                            return
                        
                        # СОРТИРУЕМ по проекции для наглядности
                        sorted_projections = sorted(projections.items(), key=lambda x: x[1])
                        
                        # Визуализация
                        st.subheader("📊 Распределение слов вдоль оси")
                        
                        # Создаем DataFrame для визуализации
                        df = pd.DataFrame([
                            {'word': word, 'projection': projection, 'abs_projection': abs(projection)}
                            for word, projection in sorted_projections
                        ])
                        
                        # Цвета в зависимости от позиции
                        df['color'] = df['projection'].apply(
                            lambda x: 'red' if x < -0.1 else 'green' if x > 0.1 else 'gray'
                        )
                        
                        # Визуализация
                        fig = px.scatter(
                            df, x='projection', y=[0]*len(df),
                            text='word', color='color',
                            title=f"Семантическая ось: '{word1}' ←→ '{word2}'",
                            labels={'projection': 'Позиция на оси', 'y': ''},
                            color_discrete_map={'red': 'red', 'green': 'green', 'gray': 'gray'}
                        )
                        
                        # Настройка внешнего вида
                        fig.update_traces(
                            marker=dict(size=15, opacity=0.7),
                            textposition='top center'
                        )
                        fig.update_layout(
                            showlegend=False,
                            yaxis=dict(showticklabels=False),
                            height=500
                        )
                        
                        # Добавляем ориентиры
                        fig.add_vline(x=-0.5, line_dash="dash", line_color="red", 
                                    annotation_text=word1, annotation_position="top left")
                        fig.add_vline(x=0.5, line_dash="dash", line_color="green",
                                    annotation_text=word2, annotation_position="top right")
                        fig.add_vline(x=0, line_dash="dot", line_color="gray")
                        
                        st.plotly_chart(fig)
                        
                        # ТАБЛИЦА РЕЗУЛЬТАТОВ
                        st.subheader("📋 Детальные результаты")
                        
                        # Создаем интерпретацию
                        interpretation_df = pd.DataFrame([
                            {
                                'Слово': word,
                                'Проекция': f"{projection:.4f}",
                                'Позиция': (
                                    f"ближе к '{word1}'" if projection < -0.1 else
                                    f"ближе к '{word2}'" if projection > 0.1 else
                                    "нейтральное"
                                ),
                                'Абс. значение': f"{abs(projection):.4f}"
                            }
                            for word, projection in sorted_projections
                        ])
                        
                        st.dataframe(interpretation_df, use_container_width=True)
                        
                        # ГРУППИРОВКА
                        st.subheader("🎯 Группировка слов")
                        
                        left_words = [word for word, proj in sorted_projections if proj < -0.1]
                        neutral_words = [word for word, proj in sorted_projections if -0.1 <= proj <= 0.1]
                        right_words = [word for word, proj in sorted_projections if proj > 0.1]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(f"Ближе к '{word1}'", len(left_words))
                            if left_words:
                                for word in left_words:
                                    st.write(f"• {word}")
                        
                        with col2:
                            st.metric("Нейтральные", len(neutral_words))
                            if neutral_words:
                                for word in neutral_words:
                                    st.write(f"• {word}")
                        
                        with col3:
                            st.metric(f"Ближе к '{word2}'", len(right_words))
                            if right_words:
                                for word in right_words:
                                    st.write(f"• {word}")
                        
                        # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
                        if show_debug:
                            st.subheader("🔧 Отладочная информация")
                            
                            # Сходства между крайними точками
                            similarity = axis_result.get('similarity_between_ends', 0)
                            st.write(f"Сходство '{word1}' и '{word2}': {similarity:.4f}")
                            
                            # Статистика оси
                            bias_analysis = axis_result.get('bias_analysis', {})
                            if bias_analysis:
                                st.write("**Статистика оси:**")
                                st.write(f"- Среднее: {bias_analysis.get('mean', 0):.4f}")
                                st.write(f"- Диапазон: {bias_analysis.get('range', 0):.4f}")
                                st.write(f"- Количество слов: {bias_analysis.get('count', 0)}")
                            
                            # Примеры ближайших соседей
                            st.write("**Ближайшие соседи:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                neighbors1 = self.semantic_analyzer.find_similar_words(self.selected_model, word1, 5)
                                if neighbors1:
                                    st.write(f"К '{word1}':")
                                    for neighbor, sim in neighbors1:
                                        st.write(f"  - {neighbor} ({sim:.3f})")
                            with col2:
                                neighbors2 = self.semantic_analyzer.find_similar_words(self.selected_model, word2, 5)
                                if neighbors2:
                                    st.write(f"К '{word2}':")
                                    for neighbor, sim in neighbors2:
                                        st.write(f"  - {neighbor} ({sim:.3f})")
                    
                    else:
                        st.error("Не удалось проанализировать ось")
                        
                except Exception as e:
                    st.error(f"Ошибка при анализе оси: {e}")
                    st.info("Попробуйте выбрать другие слова для оси")
    
    def render_visualization(self):
        """Интерфейс 2D/3D визуализации с семантическими кластерами"""
        st.header("🎨 2D/3D визуализация векторных пространств")
        
        col1, col2 = st.columns(2)
        
        with col1:
            visualization_type = st.selectbox(
                "Тип визуализации:",
                ["t-SNE", "UMAP"]
            )
            n_components = st.selectbox("Размерность:", [2, 3], index=0)
        
        with col2:
            sample_size = st.slider("Размер выборки:", 100, 1000, 300)
            perplexity = st.slider("Perplexity (t-SNE):", 5, 50, 30) if visualization_type == "t-SNE" else 15
        
        if st.button("Визуализировать") and self.selected_model:
            with st.spinner("Снижение размерности и визуализация..."):
                try:
                    # Получаем векторы слов
                    words = list(self.selected_model.wv.key_to_index.keys())[:sample_size]
                    vectors = np.array([self.selected_model.wv[word] for word in words])
                    
                    # Применяем снижение размерности
                    if visualization_type == "t-SNE":
                        embeddings = self.dimensionality_reducer.apply_tsne(
                            vectors, n_components=n_components, perplexity=perplexity
                        )
                    else:
                        embeddings = self.dimensionality_reducer.apply_umap(
                            vectors, n_components=n_components
                        )
                    
                    # Кластеризация для выделения семантических кластеров
                    n_clusters = min(10, len(vectors))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(vectors)
                    
                    # Визуализация
                    fig = self.dimensionality_reducer.visualize_embeddings(
                        embeddings, 
                        labels=words,
                        cluster_labels=[f"Кластер {label}" for label in cluster_labels]
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Информация о кластерах
                    st.subheader("📊 Информация о семантических кластерах")
                    cluster_info = []
                    for cluster_id in range(n_clusters):
                        cluster_words = [words[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                        if cluster_words:
                            cluster_info.append({
                                'Кластер': cluster_id,
                                'Количество слов': len(cluster_words),
                                'Примеры слов': ', '.join(cluster_words[:5])
                            })
                    
                    st.table(pd.DataFrame(cluster_info))
                    
                except Exception as e:
                    st.error(f"Ошибка при визуализации: {e}")
    
    def render_dynamic_report(self):
        """Генерация динамического отчета"""
        st.header("📋 Динамический отчет")
        
        if not self.selected_model:
            st.warning("Выберите модель для генерации отчета")
            return
        
        if st.button("Сгенерировать отчет"):
            with st.spinner("Генерация отчета..."):
                try:
                    # Статистика модели
                    st.subheader("📊 Статистика модели")
                    vocab_size = len(self.selected_model.wv.key_to_index)
                    vector_size = self.selected_model.wv.vector_size
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Размер словаря", vocab_size)
                    col2.metric("Размерность векторов", vector_size)
                    
                    # Точность аналогий
                    analogies_result = DistributedModels().evaluate_word_analogies(self.selected_model)
                    col3.metric("Точность аналогий", f"{analogies_result.get('accuracy', 0):.2%}")
                    
                    # Векторная арифметика - примеры
                    st.subheader("🧮 Примеры векторной арифметики")
                    test_cases = [
                        ("столица России - Москва + Франция", ["париж", "франция"], ["москва", "россия"]),
                        ("король - мужчина + женщина", ["король", "женщина"], ["мужчина"]),
                        ("холодный - лето + зима", ["холодный", "зима"], ["лето"])
                    ]
                    
                    for description, positive, negative in test_cases:
                        with st.expander(description):
                            result = self.semantic_analyzer.vector_arithmetic(
                                self.selected_model, positive, negative, topn=5
                            )
                            if result:
                                final_df = pd.DataFrame(result, 
                                                      columns=['Слово', 'Сходство'])
                                st.table(final_df)
                            else:
                                st.write("Не удалось вычислить")
                    
                    # Heatmap семантических близостей
                    st.subheader("🔥 Heatmap семантических близостей (топ-20 слов)")
                    top_words = list(self.selected_model.wv.key_to_index.keys())[:20]
                    if len(top_words) >= 2:
                        heatmap_fig = self.semantic_analyzer.create_similarity_heatmap(
                            self.selected_model, top_words
                        )
                        st.plotly_chart(heatmap_fig)
                    else:
                        st.warning("Недостаточно слов для построения heatmap")
                    
                    # 2D проекция
                    st.subheader("🎨 2D проекция векторного пространства")
                    words_sample = list(self.selected_model.wv.key_to_index.keys())[:100]
                    if len(words_sample) >= 2:
                        vectors = np.array([self.selected_model.wv[word] for word in words_sample])
                        
                        embeddings = self.dimensionality_reducer.apply_tsne(vectors, n_components=2)
                        
                        # Кластеризация для визуализации
                        n_clusters = min(5, len(vectors))
                        if n_clusters > 1:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = kmeans.fit_predict(vectors)
                            
                            fig = self.dimensionality_reducer.visualize_embeddings(
                                embeddings, 
                                labels=words_sample,
                                cluster_labels=[f"Кластер {label}" for label in cluster_labels]
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("Недостаточно данных для кластеризации")
                    else:
                        st.warning("Недостаточно слов для визуализации")
                    
                except Exception as e:
                    st.error(f"Ошибка при генерации отчета: {e}")
    
    def run(self):
        """Запуск веб-приложения"""
        st.set_page_config(
            page_title="Анализатор векторных пространств",
            page_icon="🔍",
            layout="wide"
        )
        
        page = self.render_sidebar()
        
        if page == "Векторная арифметика":
            self.render_vector_arithmetic()
        elif page == "Семантическое сходство":
            self.render_semantic_similarity()
        elif page == "Семантические оси":
            self.render_semantic_axes()
        elif page == "Визуализация":
            self.render_visualization()
        elif page == "Динамический отчет":
            self.render_dynamic_report()