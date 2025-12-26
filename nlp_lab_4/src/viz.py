from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    umap = None

from sklearn.decomposition import PCA


def project_2d(X: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    if method == "umap" and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    # fallback PCA
    return PCA(n_components=2, random_state=random_state).fit_transform(X)


def plot_clusters(Z2: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.show()
