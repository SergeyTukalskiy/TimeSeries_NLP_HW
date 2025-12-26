import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pandas as pd
from typing import List, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import networkx as nx
import random

class SemanticAnalyzer:
    """Класс для семантического анализа векторных пространств"""
    
    def __init__(self):
        pass
    
    def cosine_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Вычисление матрицы косинусного сходства"""
        return cosine_similarity(vectors)
    
    def find_similar_words(self, model, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Поиск ближайших соседей для слова"""
        try:
            return model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def vector_arithmetic(self, model, positive: List[str], negative: List[str] = None, topn: int = 10) -> List[Tuple[str, float]]:
        """Векторная арифметика"""
        try:
            return model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        except:
            return []
    
    def analyze_vector_arithmetic_steps(self, model, expression: str) -> Dict[str, Any]:
        """Анализ промежуточных шагов векторной арифметики"""
        steps = []
        current_vectors = []
        
        # Парсинг выражения типа "король - мужчина + женщина"
        parts = expression.split()
        result_vector = None
        
        for i, part in enumerate(parts):
            if part in ['+', '-']:
                continue
                
            try:
                vector = model.wv[part]
                
                if i == 0:
                    result_vector = vector
                    steps.append({
                        'step': f"Начальный вектор: {part}",
                        'vector': vector,
                        'neighbors': self.find_similar_words(model, part, 5)
                    })
                else:
                    operation = parts[i-1]
                    if operation == '+':
                        result_vector += vector
                    else:  # '-'
                        result_vector -= vector
                    
                    steps.append({
                        'step': f"{operation} {part}",
                        'vector': vector,
                        'neighbors': self.find_similar_words(model, part, 5),
                        'result_neighbors': self._find_neighbors_by_vector(model, result_vector, 10)
                    })
                    
            except KeyError:
                steps.append({
                    'step': f"Слово '{part}' не найдено в словаре",
                    'vector': None,
                    'neighbors': []
                })
        
        # Финальный результат
        if result_vector is not None:
            final_neighbors = self._find_neighbors_by_vector(model, result_vector, 10)
            steps.append({
                'step': "Финальный результат",
                'vector': result_vector,
                'neighbors': final_neighbors,
                'result_neighbors': final_neighbors
            })
        
        return {
            'steps': steps,
            'final_result': final_neighbors if result_vector is not None else []
        }
    
    def _find_neighbors_by_vector(self, model, vector: np.ndarray, topn: int = 10) -> List[Tuple[str, float]]:
        """Поиск ближайших соседей по вектору"""
        try:
            return model.wv.similar_by_vector(vector, topn=topn)
        except:
            return []
    
    def calculate_word_distance(self, model, word1: str, word2: str) -> float:
        """Вычисление косинусного расстояния между словами"""
        try:
            return model.wv.similarity(word1, word2)
        except KeyError:
            return 0.0
    
    def analyze_semantic_axes(self, model, axis_pairs: Dict[str, Tuple[str, str]], test_words: List[str]) -> Dict[str, Any]:
        """Корректный анализ семантических осей"""
        results = {}
        
        for axis_name, (word1, word2) in axis_pairs.items():
            try:
                # Получаем векторы крайних точек
                vec1 = model.wv[word1]
                vec2 = model.wv[word2]
                
                # ВЫЧИСЛЯЕМ ОСЬ КОРРЕКТНО: нормализованное направление
                axis_direction = vec2 - vec1
                axis_norm = np.linalg.norm(axis_direction)
                
                if axis_norm < 1e-10:  # Избегаем деления на ноль
                    continue
                    
                axis_direction_normalized = axis_direction / axis_norm
                
                # Вычисляем проекции для всех тестовых слов
                projections = {}
                for word in test_words:
                    try:
                        word_vec = model.wv[word]
                        
                        # ПРОЕКЦИЯ: расстояние от слова до оси
                        # Сначала находим проекцию на ось
                        projection_on_axis = np.dot(word_vec - vec1, axis_direction_normalized)
                        
                        # Нормализуем проекцию относительно длины оси
                        normalized_projection = projection_on_axis / axis_norm
                        
                        projections[word] = normalized_projection
                        
                    except KeyError:
                        continue  # Слово не в словаре
                
                # Вычисляем проекции для самих крайних точек для калибровки
                projection_word1 = np.dot(vec1 - vec1, axis_direction_normalized) / axis_norm  # Должно быть ~0
                projection_word2 = np.dot(vec2 - vec1, axis_direction_normalized) / axis_norm  # Должно быть ~1
                
                # Анализ смещения
                bias_analysis = self._analyze_bias_on_axis(projections)
                
                results[axis_name] = {
                    'axis_direction': axis_direction,
                    'axis_length': axis_norm,
                    'projections': projections,
                    'bias_analysis': bias_analysis,
                    'reference_projections': {
                        word1: projection_word1,  # ~0
                        word2: projection_word2   # ~1
                    }
                }
                
            except KeyError as e:
                print(f"Слово оси не найдено в словаре: {e}")
                continue
        
        return results

    def analyze_semantic_axes_advanced(self, model, axis_pairs: Dict[str, Tuple[str, str]], test_words: List[str]) -> Dict[str, Any]:
        """Улучшенный анализ с несколькими методами"""
        results = {}
        
        for axis_name, (word1, word2) in axis_pairs.items():
            try:
                vec1 = model.wv[word1]
                vec2 = model.wv[word2]
                
                # МЕТОД: Косинусное сходство (самый стабильный)
                projections_cosine = {}
                
                for word in test_words:
                    try:
                        word_vec = model.wv[word]
                        
                        # Вычисляем сходство с обеими крайними точками
                        similarity_to_word1 = cosine_similarity([word_vec], [vec1])[0][0]
                        similarity_to_word2 = cosine_similarity([word_vec], [vec2])[0][0]
                        
                        # Проекция как разность сходств (более интуитивно)
                        # Отрицательное: ближе к word1, Положительное: ближе к word2
                        projection = similarity_to_word2 - similarity_to_word1
                        projections_cosine[word] = projection
                            
                    except KeyError:
                        continue
                
                # Анализ смещения
                bias_analysis = self._analyze_bias_on_axis(projections_cosine)
                
                # Вычисляем сходство между крайними точками для отладки
                similarity_between_ends = cosine_similarity([vec1], [vec2])[0][0]
                
                results[axis_name] = {
                    'projections': projections_cosine,  # Основной метод
                    'bias_analysis': bias_analysis,
                    'similarity_between_ends': similarity_between_ends,
                    'vector_norms': {
                        'norm_word1': np.linalg.norm(vec1),
                        'norm_word2': np.linalg.norm(vec2)
                    }
                }
                
            except KeyError as e:
                print(f"Слово оси не найдено: {e}")
                continue
        
        return results
    
    def _analyze_bias_on_axis(self, projections: Dict[str, float]) -> Dict[str, Any]:
        """Анализ смещения на семантической оси"""
        if not projections:
            return {}
        
        values = list(projections.values())
        
        # Нормализуем значения для лучшей интерпретации
        if len(values) > 1:
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val != min_val else 1
            normalized_values = [(v - min_val) / range_val for v in values]
        else:
            normalized_values = values
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'normalized_mean': np.mean(normalized_values),
            'bias_score': np.std(normalized_values),  # Мера смещения на нормализованной шкале
            'range': max(values) - min(values),
            'count': len(values)
        }
    
    def create_semantic_network(self, model, seed_words: List[str], depth: int = 2, threshold: float = 0.5) -> Dict[str, Any]:
        """Создание семантической сети с графами"""
        G = nx.Graph()
        visited = set()
        to_visit = [(word, 0) for word in seed_words]
        
        while to_visit:
            current_word, current_depth = to_visit.pop(0)
            
            if current_word in visited or current_depth > depth:
                continue
            
            visited.add(current_word)
            # Сохраняем глубину в атрибутах узла
            G.add_node(current_word, depth=current_depth)
            
            # Находим соседей
            neighbors = self.find_similar_words(model, current_word, topn=10)
            for neighbor, similarity in neighbors:
                if similarity >= threshold:
                    G.add_edge(current_word, neighbor, weight=similarity)
                    
                    if neighbor not in visited:
                        to_visit.append((neighbor, current_depth + 1))
        
        # Конвертируем в формат для визуализации
        nodes = []
        for node in G.nodes():
            node_data = {'id': node, 'group': G.nodes[node].get('depth', 0)}
            nodes.append(node_data)
            
        links = []
        for u, v, data in G.edges(data=True):
            link_data = {'source': u, 'target': v, 'value': data.get('weight', 0)}
            links.append(link_data)
        
        return {
            'graph': G,
            'nodes': nodes,
            'links': links,
            'metrics': {
                'nodes_count': len(nodes),
                'edges_count': len(links),
                'density': nx.density(G) if len(nodes) > 1 else 0
            }
        }
    
    def analyze_distance_distribution(self, model, sample_words: List[str] = None, max_pairs: int = 5000) -> Dict[str, Any]:
        """Анализ распределения расстояний в пространстве"""
        if sample_words is None:
            # Берем случайную выборку слов для анализа
            all_words = list(model.wv.key_to_index.keys())
            sample_size = min(100, len(all_words))  # Ограничиваем размер выборки
            sample_words = all_words[:sample_size]
        
        distances = []
        n_pairs = 0
        
        # Ограничиваем количество пар для вычисления
        for i, word1 in enumerate(sample_words):
            for j, word2 in enumerate(sample_words):
                # ИСКЛЮЧАЕМ сравнение слов с самими собой
                if i < j and n_pairs < max_pairs:
                    try:
                        dist = self.calculate_word_distance(model, word1, word2)
                        distances.append(dist)
                        n_pairs += 1
                    except:
                        continue
        
        if not distances:
            return {
                'distances': [],
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'n_pairs': 0,
                'sample_size': len(sample_words)
            }
        
        return {
            'distances': distances,
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': min(distances),
            'max': max(distances),
            'n_pairs': n_pairs,
            'sample_size': len(sample_words)
        }
    
    def create_similarity_heatmap(self, model, words: List[str]) -> go.Figure:
        """Создание heatmap семантических близостей"""
        n = len(words)
        similarity_matrix = np.zeros((n, n))
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                similarity_matrix[i, j] = self.calculate_word_distance(model, word1, word2)
        
        fig = px.imshow(similarity_matrix,
                       x=words,
                       y=words,
                       title="Heatmap семантических близостей",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        
        return fig
    
    def evaluate_clustering_quality(self, document_vectors: np.ndarray, 
                                  true_labels: List[str] = None) -> Dict[str, float]:
        """Оценка качества кластеризации"""
        if true_labels is None:
            # Если истинные метки отсутствуют, используем силуэтный коэффициент
            if len(document_vectors) > 1:
                n_clusters = min(5, len(document_vectors))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(document_vectors)
                    silhouette = silhouette_score(document_vectors, cluster_labels)
                    return {
                        'silhouette_score': silhouette,
                        'n_clusters': n_clusters
                    }
            return {}
        
        # Используем K-means для кластеризации
        n_clusters = len(set(true_labels))
        if n_clusters > 1 and len(document_vectors) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            predicted_labels = kmeans.fit_predict(document_vectors)
            
            # Adjusted Rand Index
            ari = adjusted_rand_score(true_labels, predicted_labels)
            silhouette = silhouette_score(document_vectors, predicted_labels)
            
            return {
                'adjusted_rand_index': ari,
                'silhouette_score': silhouette,
                'n_clusters': n_clusters
            }
        
        return {}