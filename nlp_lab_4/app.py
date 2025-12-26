import streamlit as st
import numpy as np
import pandas as pd
import json

from src.text_preprocessing import preprocess_and_tokenize, PreprocessConfig
from src.text_to_vector import TextVectorizer, VectorConfig
from src.clustering import clusterize, ClusterConfig
from src.evaluate import evaluate_clustering
from src.viz import project_2d

import matplotlib.pyplot as plt


def load_jsonl(path: str, text_field: str = "text"):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj[text_field])
            if "label" in obj:
                labels.append(obj["label"])
    y = np.array(labels) if labels else None
    return texts, y


st.title("Text Clustering Lab")

corpus_path = st.text_input("Путь к корпусу (jsonl)", "data/corpus.jsonl")
bpe_codes = st.text_input("Путь к BPE codes (если выбран BPE)", "models/bpe.codes")

tok_mode = st.selectbox("Токенизация", ["whitespace", "regex", "bpe"])
vec_method = st.selectbox("Векторизация", ["tfidf", "bm25", "w2v", "ft", "glove"])
cluster_method = st.selectbox("Кластеризация", [
    "kmeans", "minibatch_kmeans", "spherical_kmeans",
    "dbscan", "hdbscan",
    "agglomerative", "gmm", "spectral"
])

n_clusters = st.slider("k / num_topics / n_components", 2, 50, 12)

run_btn = st.button("Запустить")

if run_btn:
    texts, y = load_jsonl(corpus_path)
    st.write(f"Документов: {len(texts)}")

    pre_cfg = PreprocessConfig(lower=True, lemmatize=True, min_token_len=2)
    tokens_list = preprocess_and_tokenize(texts, pre_cfg, tok_mode, bpe_codes_path=(bpe_codes if tok_mode == "bpe" else None))

    vcfg = VectorConfig(method=vec_method, use_l2_norm=True)
    vec = TextVectorizer(vcfg)

    if vec_method in ["w2v", "ft", "glove"]:
        path = st.text_input("Путь к модели эмбеддингов", "models/w2v_cbow.model")
        kind = "w2v" if vec_method == "w2v" else ("ft" if vec_method == "ft" else "glove")
        vec.load_embeddings(path, kind=kind)

    if vec_method in ["tfidf", "bm25"]:
        vec.fit(tokens_list)

    X = vec.transform(tokens_list)

    ccfg = ClusterConfig(method=cluster_method, n_clusters=n_clusters, n_components=n_clusters, num_topics=n_clusters) if False else ClusterConfig(method=cluster_method, n_clusters=n_clusters, n_components=n_clusters)
    labels, meta = clusterize(X, ccfg)

    metrics = evaluate_clustering(X, labels, y_true=y)
    st.subheader("Метрики")
    st.json(metrics)

    st.subheader("Визуализация")
    Z2 = project_2d(X, method="umap")
    fig = plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels)
    plt.title(f"{vec_method}+{cluster_method}")
    st.pyplot(fig)

    st.subheader("Просмотр кластеров")
    df = pd.DataFrame({"text": texts, "cluster": labels})
    chosen = st.selectbox("Кластер", sorted(df["cluster"].unique().tolist()))
    st.dataframe(df[df["cluster"] == chosen].head(50))
