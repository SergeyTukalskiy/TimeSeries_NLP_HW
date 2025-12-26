from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np

def top_terms_tfidf(X_tfidf: np.ndarray, labels: np.ndarray, feature_names: List[str], top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    out: Dict[int, List[Tuple[str, float]]] = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_vec = X_tfidf[mask].mean(axis=0)
        idx = np.argsort(mean_vec)[::-1][:top_n]
        out[c] = [(feature_names[i], float(mean_vec[i])) for i in idx]
    return out


def nearest_words_to_centroid(emb, X: np.ndarray, labels: np.ndarray, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    """
    emb: gensim KeyedVectors
    X: doc vectors (L2 normalized ok)
    """
    out = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        mask = labels == c
        if mask.sum() == 0:
            continue
        centroid = X[mask].mean(axis=0)
        # gensim: similar_by_vector
        out[c] = [(w, float(s)) for w, s in emb.similar_by_vector(centroid, topn=top_n)]
    return out
