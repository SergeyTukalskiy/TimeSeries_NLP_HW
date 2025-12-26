import os
import sys
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.classical_classifiers import train_and_eval_classical
from src.data_utils import load_jsonl
from src.text_preprocessing import basic_clean, preprocess_corpus


DATA_TRAIN_PATH = "data/splits/train.jsonl"
DATA_VALID_PATH = "data/splits/valid.jsonl"


# ======================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def prepare_xy(path: str, label_field: str):
    docs = load_jsonl(path)
    X = preprocess_corpus(docs)              # "title. text" -> очищенный текст
    y = [d[label_field] for d in docs]
    return X, y, docs


@st.cache_resource
def train_models_for_task(label_field: str, task_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Тренируем классические модели для указанной задачи.
    Кэшируем результат, чтобы не обучать на каждом перезапуске Streamlit.
    Возвращаем словарь:
    {
      model_name: {
         "pipeline": sklearn Pipeline(TFIDF + модель),
         "accuracy": ...,
         "f1_macro": ...,
      },
      ...
    }
    """
    X_train, y_train, _ = prepare_xy(DATA_TRAIN_PATH, label_field)
    X_valid, y_valid, _ = prepare_xy(DATA_VALID_PATH, label_field)

    # используем уже написанную функцию из classical_classifiers
    results = train_and_eval_classical(
        X_train, y_train, X_valid, y_valid, task_name=task_name
    )

    # дополнительно посчитаем и сохраним classification_report в текстовом виде
    for name, info in results.items():
        pipe: Pipeline = info["pipeline"]
        y_pred = pipe.predict(X_valid)
        report = classification_report(y_valid, y_pred, digits=3)
        info["report"] = report

    return results


def predict_with_proba(model: Pipeline, text: str):
    """
    Обёртка для предсказания:
    - text: сырой текст пользователя
    - model: TFIDF + classifier
    Возвращаем: метка, вероятности (если есть).
    """
    X = [basic_clean(text)]
    clf = model
    y_pred = clf.predict(X)[0]

    proba_dict = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        classes = clf.classes_
        proba_dict = dict(zip(classes, proba))

    return y_pred, proba_dict


# ======================
#  STREAMLIT UI
# ======================

def main():
    st.set_page_config(
        page_title="Классификация новостей РБК",
        layout="wide"
    )

    st.title("🔎 Лабораторная: классификация текстов (корпус РБК)")

    st.markdown(
        """
        Это интерактивный интерфейс к классическим моделям классификации текста,
        обученным на корпусе `rbc_articles_words.jsonl` (новости РБК).

        В этом приложении:
        - можно выбрать **тип задачи** (бинарная / многоклассовая),
        - выбрать **модель** (логистическая регрессия, линейный SVM, случайный лес),
        - ввести произвольный текст новости и получить предсказание модели.
        """
    )

    # ---- боковая панель ----
    st.sidebar.header("⚙️ Настройки эксперимента")

    task = st.sidebar.radio(
        "Тип задачи:",
        options=["Бинарная (sentiment)", "Многоклассовая (category)"],
        index=1,
    )

    if task.startswith("Бинарная"):
        label_field = "sentiment"
        task_name = "binary"
    else:
        label_field = "category"
        task_name = "multiclass"

    st.sidebar.write("---")
    st.sidebar.subheader("Модели")

    # обучаем/загружаем модели для выбранной задачи (кэшируется)
    with st.spinner("Обучение/загрузка моделей..."):
        models_dict = train_models_for_task(label_field, task_name)

    model_names = list(models_dict.keys())
    selected_model_name = st.sidebar.selectbox(
        "Выберите модель:",
        options=model_names,
        index=0
    )

    selected_model_info = models_dict[selected_model_name]
    selected_pipeline: Pipeline = selected_model_info["pipeline"]

    # ---- основная часть: метрики ----
    st.subheader("📊 Качество модели на валидации")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{selected_model_info['accuracy']:.3f}")
    with col2:
        st.metric("F1-macro", f"{selected_model_info['f1_macro']:.3f}")

    with st.expander("Показать подробный classification_report"):
        st.text(selected_model_info["report"])

    st.markdown("---")

    # ---- ввод текста пользователем ----
    st.subheader("📝 Классификация пользовательского текста")

    default_text = (
        "Банк России принял решение снизить ключевую ставку, "
        "что привело к росту курса акций крупнейших российских компаний."
        if task_name == "multiclass"
        else "Российский рынок акций продолжает укрепляться на фоне роста цен на нефть."
    )

    user_text = st.text_area(
        "Введите текст новости или короткое описание:",
        value=default_text,
        height=200
    )

    if st.button("🔮 Классифицировать"):
        if not user_text.strip():
            st.warning("Введите какой-нибудь текст для классификации.")
        else:
            pred_label, proba_dict = predict_with_proba(selected_pipeline, user_text)

            st.write("### Результат предсказания:")
            if task_name == "binary":
                label_human = "экономическая / про рынок (1)" if pred_label == 1 else "другая тематика (0)"
                st.markdown(f"**Класс (sentiment):** `{pred_label}` — *{label_human}*")
            else:
                st.markdown(f"**Класс (category):** `{pred_label}`")

            if proba_dict is not None:
                st.write("#### Вероятности по классам:")
                proba_df = (
                    pd.DataFrame(
                        {
                            "Класс": list(proba_dict.keys()),
                            "Вероятность": list(proba_dict.values())
                        }
                    )
                    .sort_values("Вероятность", ascending=False)
                )
                st.dataframe(proba_df, use_container_width=True)
            else:
                st.info(
                    "Выбранная модель не поддерживает `predict_proba` "
                    "(например, LinearSVC). Для вероятностей используйте логистическую регрессию или случайный лес."
                )

    st.markdown("---")

    # ---- просмотр примеров из валидационной выборки ----
    st.subheader("🔍 Примеры из валидационной выборки")

    _, _, docs_valid = prepare_xy(DATA_VALID_PATH, label_field)
    n_examples = st.slider("Сколько примеров показать:", 3, 20, 5)

    # получаем предсказания для нескольких примеров
    texts_valid = [
        f"{d.get('title', '')}. {d.get('text', '')}" for d in docs_valid[:n_examples]
    ]
    cleaned_valid = [basic_clean(t) for t in texts_valid]
    y_true = [d[label_field] for d in docs_valid[:n_examples]]
    y_pred = selected_pipeline.predict(cleaned_valid)

    for i in range(n_examples):
        with st.expander(f"Пример {i+1}"):
            st.markdown(f"**Заголовок:** {docs_valid[i].get('title', '')}")
            st.markdown(f"**Текст (укороченный):** {docs_valid[i].get('text', '')[:500]}...")
            st.markdown(f"**Истинный класс:** `{y_true[i]}`")
            st.markdown(f"**Предсказанный класс ({selected_model_name}):** `{y_pred[i]}`")


if __name__ == "__main__":
    main()
