from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

# HDBSCAN optional
try:
    import hdbscan
except Exception:
    hdbscan = None


@dataclass
class ClusterConfig:
    method: str
    # common
    n_clusters: int = 10

    # kmeans / minibatch
    random_state: int = 42
    batch_size: int = 2048

    # dbscan/hdbscan
    eps: float = 0.3
    min_samples: int = 10

    # agglomerative
    linkage: str = "average"  # 'ward'|'single'|'complete'|'average'
    metric: str = "cosine"    # 'cosine' or 'euclidean' etc.

    # gmm
    n_components: int = 10
    covariance_type: str = "diag"

    # spectral
    affinity: str = "nearest_neighbors"
    n_neighbors: int = 20


class SphericalKMeans:
    """
    Простая реализация spherical k-means:
    - вход X должен быть L2-нормирован
    - assignment по max cosine similarity = max dot(X, C)
    - центроиды пересчитываем как mean и затем нормируем
    """
    def __init__(self, n_clusters: int, n_init: int = 5, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        best_inertia = float("inf")
        best_labels = None
        best_centers = None

        for _ in range(self.n_init):
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            C = X[idx].copy()

            for _it in range(self.max_iter):
                # cosine sim on normalized => dot product
                sims = X @ C.T
                labels = sims.argmax(axis=1)

                newC = np.zeros_like(C)
                for k in range(self.n_clusters):
                    members = X[labels == k]
                    if len(members) == 0:
                        newC[k] = X[rng.integers(0, X.shape[0])]
                    else:
                        v = members.mean(axis=0)
                        n = np.linalg.norm(v) + 1e-12
                        newC[k] = v / n

                if np.allclose(newC, C, atol=1e-5):
                    C = newC
                    break
                C = newC

            # inertia surrogate: sum(1 - cos) = sum(1 - max sim)
            sims = (X @ C.T).max(axis=1)
            inertia = float(np.sum(1.0 - sims))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = C

        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        return self


def clusterize(X: np.ndarray, cfg: ClusterConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": cfg.method}

    if cfg.method == "kmeans":
        model = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init="auto")
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
            batch_size=cfg.batch_size,
            n_init="auto"
        )
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "spherical_kmeans":
        # ожидание: X уже L2-нормирован
        model = SphericalKMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state)
        model.fit(X)
        meta["model"] = model
        return model.labels_, meta

    if cfg.method == "dbscan":
        model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, metric=cfg.metric)
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan не установлен. Установи: pip install hdbscan")
        model = hdbscan.HDBSCAN(min_samples=cfg.min_samples)
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=cfg.n_clusters,
            linkage=cfg.linkage,
            metric=("euclidean" if cfg.linkage == "ward" else cfg.metric),
        )
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "gmm":
        model = GaussianMixture(
            n_components=cfg.n_components,
            covariance_type=cfg.covariance_type,
            random_state=cfg.random_state
        )
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    if cfg.method == "spectral":
        model = SpectralClustering(
            n_clusters=cfg.n_clusters,
            affinity=cfg.affinity,
            n_neighbors=cfg.n_neighbors,
            random_state=cfg.random_state
        )
        labels = model.fit_predict(X)
        meta["model"] = model
        return labels, meta

    raise ValueError(f"Unknown clustering method: {cfg.method}")
