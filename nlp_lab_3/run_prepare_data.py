from src.data_utils import (
    create_labeled_corpus,
    load_jsonl,
    stratified_split,
    save_jsonl
)

if __name__ == "__main__":
    input_path = "data/rbc_articles_words.jsonl"   # <<< ВАЖНО
    labeled_path = "data/rbc_labeled.jsonl"

    # 1) добавляем sentiment + categories, но не ломаем твои поля
    create_labeled_corpus(input_path, labeled_path, max_docs=None)

    # 2) делаем сплиты по многоклассовой задаче category
    docs = load_jsonl(labeled_path)
    train_docs, valid_docs, test_docs = stratified_split(
        docs, label_field="category"
    )

    save_jsonl(train_docs, "data/splits/train.jsonl")
    save_jsonl(valid_docs, "data/splits/valid.jsonl")
    save_jsonl(test_docs, "data/splits/test.jsonl")

    print(
        f"Train: {len(train_docs)}, "
        f"Valid: {len(valid_docs)}, "
        f"Test: {len(test_docs)}"
    )
