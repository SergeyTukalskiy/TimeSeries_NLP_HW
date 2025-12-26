from typing import Dict, Any, Tuple, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)


def build_tfidf_vectorizer(
    ngram_range=(1, 2),
    max_features: int | None = 50000
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )


def get_classical_models(num_classes: int) -> Dict[str, Any]:
    """
    Возвращает словарь классических моделей.
    XGBoost временно отключен, чтобы не мучиться с кодированием строковых меток.
    Для лабы достаточно логрегрессии, линейного SVM и случайного леса.
    """
    models: Dict[str, Any] = {
        "logreg_l2": LogisticRegression(
            penalty="l2",
            max_iter=500,
            n_jobs=-1
        ),
        "linear_svm": LinearSVC(),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1
        ),
    }

    # Если захочешь вернуть XGBoost, сюда вернём ещё одну модель
    # с кодированием меток через LabelEncoder.

    return models


def train_and_eval_classical(
    X_train: List[str],
    y_train: List[Any],
    X_valid: List[str],
    y_valid: List[Any],
    task_name: str = "multiclass"
) -> Dict[str, Dict[str, Any]]:
    """
    Строим TF-IDF + модель, считаем метрики на валидации.
    Возвращаем отчёты по моделям.
    """
    num_classes = len(set(y_train))
    vectorizer = build_tfidf_vectorizer()

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in get_classical_models(num_classes).items():
        print(f"\n=== Training {name} ({task_name}) ===")

        clf = Pipeline([
            ("tfidf", vectorizer),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_valid)

        acc = accuracy_score(y_valid, y_pred)
        f1_macro = f1_score(y_valid, y_pred, average="macro")

        print(f"Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}")
        print(classification_report(y_valid, y_pred))

        results[name] = {
            "pipeline": clf,
            "accuracy": acc,
            "f1_macro": f1_macro,
        }

    return results
