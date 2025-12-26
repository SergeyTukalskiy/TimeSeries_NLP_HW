import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go

class DimensionalityReducer:
    """Класс для снижения размерности и тематического моделирования"""
    
    def __init__(self):
        self.models = {}
        
    def apply_svd(self, matrix, n_components: int = 100) -> Dict[str, Any]:
        """Применение SVD для снижения размерности"""
        print(f"Применение SVD с {n_components} компонентами...")
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(matrix)
        
        explained_variance = svd.explained_variance_ratio_.sum()
        
        self.models['svd'] = svd
        
        return {
            'reduced_matrix': reduced_matrix,
            'components': svd.components_,
            'explained_variance': explained_variance,
            'singular_values': svd.singular_values_,
            'model': svd
        }
    
    def find_optimal_components(self, matrix, max_components: int = 200) -> Dict[str, Any]:
        """Поиск оптимального числа компонент"""
        print("Поиск оптимального числа компонент...")
        
        explained_variances = []
        components_range = range(10, max_components + 1, 10)
        
        for n_comp in components_range:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(matrix)
            explained_variances.append(svd.explained_variance_ratio_.sum())
            
        optimal_idx = np.argmax(np.array(explained_variances) >= 0.8)
        optimal_components = components_range[optimal_idx] if optimal_idx < len(components_range) else max_components
        
        return {
            'components_range': list(components_range),
            'explained_variances': explained_variances,
            'optimal_components': optimal_components
        }
    
    def visualize_components(self, svd_result: Dict[str, Any], feature_names: List[str], n_top_features: int = 10):
        """Визуализация компонент SVD"""
        components = svd_result['components']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(min(6, components.shape[0])):
            top_indices = np.argsort(np.abs(components[i]))[-n_top_features:]
            top_features = [feature_names[idx] for idx in top_indices]
            top_weights = components[i][top_indices]
            
            axes[i].barh(range(n_top_features), top_weights)
            axes[i].set_yticks(range(n_top_features))
            axes[i].set_yticklabels(top_features)
            axes[i].set_title(f'Компонента {i+1}')
        
        plt.tight_layout()
        return fig
    
    def apply_tsne(self, matrix, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """Применение t-SNE для визуализации"""
        print("Применение t-SNE...")
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(matrix)
        
        return embedded
    
    def apply_umap(self, matrix, n_components: int = 2, n_neighbors: int = 15) -> np.ndarray:
        """Применение UMAP для визуализации"""
        print("Применение UMAP...")
        
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        embedded = reducer.fit_transform(matrix)
        
        return embedded
    
    def visualize_embeddings(self, embeddings: np.ndarray, labels: List[str] = None, 
                           titles: List[str] = None, cluster_labels: List[str] = None) -> go.Figure:
        """Визуализация embeddings в 2D/3D с выделением семантических кластеров"""
        
        if embeddings.shape[1] == 2:
            if cluster_labels is not None:
                fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], 
                               color=cluster_labels, title="2D визуализация документов с кластерами",
                               hover_data={'label': labels} if labels else None)
            elif labels is not None:
                fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], 
                               color=labels, title="2D визуализация документов")
            else:
                fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1],
                               title="2D визуализация документов")
        else:
            if cluster_labels is not None:
                fig = px.scatter_3d(x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                                  color=cluster_labels, title="3D визуализация документов с кластерами",
                                  hover_data={'label': labels} if labels else None)
            elif labels is not None:
                fig = px.scatter_3d(x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                                  color=labels, title="3D визуализация документов")
            else:
                fig = px.scatter_3d(x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                                  title="3D визуализация документов")
        
        return fig