from src.data_utils import load_jsonl
from src.text_preprocessing import preprocess_corpus
from src.classical_classifiers import train_and_eval_classical


def prepare_xy(path: str, label_field: str):
    docs = load_jsonl(path)
    X = preprocess_corpus(docs)
    y = [d[label_field] for d in docs]
    return X, y


if __name__ == "__main__":
    # многоклассовая задача: category
    train_path = "data/splits/train.jsonl"
    valid_path = "data/splits/valid.jsonl"

    X_train, y_train = prepare_xy(train_path, label_field="category")
    X_valid, y_valid = prepare_xy(valid_path, label_field="category")

    results_multiclass = train_and_eval_classical(
        X_train, y_train, X_valid, y_valid, task_name="multiclass"
    )

    # бинарная задача: sentiment
    X_train_bin, y_train_bin = prepare_xy(train_path, label_field="sentiment")
    X_valid_bin, y_valid_bin = prepare_xy(valid_path, label_field="sentiment")

    results_binary = train_and_eval_classical(
        X_train_bin, y_train_bin, X_valid_bin, y_valid_bin, task_name="binary"
    )

    # здесь можно сохранить метрики в json / csv, чтобы потом вставить в отчёт
