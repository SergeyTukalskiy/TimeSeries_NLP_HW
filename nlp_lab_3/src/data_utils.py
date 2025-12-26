import json
import os
from collections import Counter
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Читает JSONL-файл:
    по одной JSON-записи в строке.
    """
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def save_jsonl(docs: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def simple_keyword_based_sentiment_and_multilabel(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    НЕ трогаем существующие поля category/date/url,
    а только добавляем: sentiment и categories.

    sentiment:
        1 — условно "экономическая" новость
        0 — всё остальное

    categories:
        список тематик (multilabel) на основе keywords + исходной category.
    """
    text = (
        (doc.get("title") or "") + " " +
        (doc.get("text") or "")
    ).lower()

    economy_kw = [
        "рубл", "доллар", "валют", "курс", "бирж", "акци", "облигац",
        "рынок", "инфляци", "ставк", "банк", "цб", "центробанк", "налог",
        "бюджет", "кредит", "ипотек", "инвест"
    ]
    politics_kw = [
        "президент", "правительств", "госдум", "сенат", "выбор", "санкц",
        "мид", "министерств", "парламент", "политик"
    ]
    sport_kw = [
        "матч", "футбол", "хоккей", "турнир", "лига", "чемпионат",
        "гол", "тренер", "игрок"
    ]
    tech_kw = [
        "технолог", " it ", "айти", "стартап", "компани", "сервис", "платформ",
        "приложен", "смартфон", "гаджет", "интернет", "искусственный интеллект",
        "нейросет"
    ]

    # --- бинарка: экономические vs остальные ---
    sentiment = 1 if any(k in text for k in economy_kw) else 0

    # --- мультиметка: собираем темы из keywords ---
    cats = set()

    if any(k in text for k in economy_kw):
        cats.add("economy")
    if any(k in text for k in politics_kw):
        cats.add("politics")
    if any(k in text for k in sport_kw):
        cats.add("sport")
    if any(k in text for k in tech_kw):
        cats.add("tech")

    # дополнительно учтём исходную category, если она есть
    orig_cat = (doc.get("category") or "").strip()
    if orig_cat:
        cats.add(orig_cat)

    if not cats:
        cats.add("society")  # мусорный / общий класс

    new_doc = {
        # сохраняем исходные поля
        "title": doc.get("title"),
        "text": doc.get("text"),
        "category": orig_cat if orig_cat else list(cats)[0],  # для мультикласса
        "date": doc.get("date"),
        "url": doc.get("url"),

        # добавляем новые поля
        "sentiment": sentiment,
        "categories": list(cats),      # для мульти-лейбла
        "orig_category": orig_cat      # на всякий случай
    }

    return new_doc


def create_labeled_corpus(
    input_path: str,
    output_path: str,
    max_docs: int | None = None
):
    # тут уже читаем JSONL, а не JSON
    data = load_jsonl(input_path)
    if max_docs is not None:
        data = data[:max_docs]

    labeled: List[Dict[str, Any]] = []

    for doc in data:
        # пропустим записи, у которых нет текста/тайтла
        if not doc.get("title") and not doc.get("text"):
            continue
        labeled_doc = simple_keyword_based_sentiment_and_multilabel(doc)
        # на всякий случай не берём записи без category (для стратификации)
        if not labeled_doc.get("category"):
            continue
        labeled.append(labeled_doc)

    save_jsonl(labeled, output_path)
    print(f"Saved {len(labeled)} labeled docs to {output_path}")


def stratified_split(
    docs: List[Dict[str, Any]],
    label_field: str,
    train_size: float = 0.7,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    min_count: int = 3,
):
    """
    Стратифицированный train/valid/test по полю label_field.
    - Сначала выбрасываем документы с редкими классами (меньше min_count).
    - Если даже после этого stratify не удаётся, делаем обычный random split
      (без стратификации), чтобы код не падал.
    """
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6

    # 1) выкидываем документы без метки
    docs_with_label = [d for d in docs if d.get(label_field) not in (None, "")]
    if len(docs_with_label) < len(docs):
        print(
            f"WARNING: dropped {len(docs)-len(docs_with_label)} docs "
            f"without label '{label_field}'"
        )

    labels = [d[label_field] for d in docs_with_label]
    counts = Counter(labels)

    # 2) оставляем только классы с частотой >= min_count
    allowed_labels = {lbl for lbl, c in counts.items() if c >= min_count}
    filtered_docs = [d for d in docs_with_label if d[label_field] in allowed_labels]
    filtered_labels = [d[label_field] for d in filtered_docs]

    dropped_for_rare = len(docs_with_label) - len(filtered_docs)
    if dropped_for_rare > 0:
        print(
            f"WARNING: dropped {dropped_for_rare} docs with rare labels "
            f"(min_count={min_count}) for '{label_field}'"
        )

    if len(filtered_docs) == 0:
        raise ValueError(
            f"No documents left for label '{label_field}' after filtering rare classes "
            f"(min_count={min_count})."
        )

    if len(set(filtered_labels)) < 2:
        print(
            f"WARNING: only one class left for '{label_field}' after filtering. "
            f"Falling back to non-stratified split."
        )
        # просто случайно делим без стратификации
        train_docs, temp_docs = train_test_split(
            filtered_docs,
            train_size=train_size,
            shuffle=True,
            random_state=random_state,
        )
        valid_rel = valid_size / (valid_size + test_size)
        valid_docs, test_docs = train_test_split(
            temp_docs,
            train_size=valid_rel,
            shuffle=True,
            random_state=random_state,
        )
        return train_docs, valid_docs, test_docs

    # 3) пытаемся сделать настоящий stratified split
    try:
        train_docs, temp_docs, y_train, y_temp = train_test_split(
            filtered_docs,
            filtered_labels,
            train_size=train_size,
            stratify=filtered_labels,
            random_state=random_state,
        )

        valid_rel = valid_size / (valid_size + test_size)

        valid_docs, test_docs, _, _ = train_test_split(
            temp_docs,
            y_temp,
            train_size=valid_rel,
            stratify=y_temp,
            random_state=random_state,
        )
        return train_docs, valid_docs, test_docs

    except ValueError as e:
        print(
            f"WARNING: stratified split for '{label_field}' failed: {e}\n"
            f"Falling back to non-stratified split."
        )
        # fallback: без stratify, но с теми же пропорциями
        train_docs, temp_docs = train_test_split(
            filtered_docs,
            train_size=train_size,
            shuffle=True,
            random_state=random_state,
        )
        valid_rel = valid_size / (valid_size + test_size)
        valid_docs, test_docs = train_test_split(
            temp_docs,
            train_size=valid_rel,
            shuffle=True,
            random_state=random_state,
        )
        return train_docs, valid_docs, test_docs
