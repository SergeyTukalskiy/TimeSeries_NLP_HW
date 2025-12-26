from __future__ import annotations
from typing import Optional, Dict
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)


def _valid_for_internal(labels: np.ndarray) -> bool:
    # внутренние метрики требуют >=2 кластеров и не все точки шум
    uniq = set(labels.tolist())
    if len(uniq) < 2:
        return False
    # если все -1 (шум) — тоже нельзя
    if uniq == {-1}:
        return False
    return True


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    res: Dict[str, float] = {}
    if _valid_for_internal(labels):
        # silhouette часто считают без шума; можно убрать -1
        mask = labels != -1
        if mask.sum() >= 2 and len(set(labels[mask].tolist())) >= 2:
            res["silhouette"] = float(silhouette_score(X[mask], labels[mask], metric="cosine"))
        else:
            res["silhouette"] = float("nan")

        res["calinski_harabasz"] = float(calinski_harabasz_score(X[labels != -1], labels[labels != -1]))
        res["davies_bouldin"] = float(davies_bouldin_score(X[labels != -1], labels[labels != -1]))
    else:
        res["silhouette"] = float("nan")
        res["calinski_harabasz"] = float("nan")
        res["davies_bouldin"] = float("nan")

    if y_true is not None:
        # внешние метрики допускают -1, но лучше сравнивать по mask, если y_true тоже есть для всех
        res["ARI"] = float(adjusted_rand_score(y_true, labels))
        res["NMI"] = float(normalized_mutual_info_score(y_true, labels))
        res["V_measure"] = float(v_measure_score(y_true, labels))

    return res
