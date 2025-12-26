from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from text_preprocessing import preprocess_and_tokenize, PreprocessConfig
from text_to_vector import TextVectorizer, VectorConfig
from clustering import clusterize, ClusterConfig
from evaluate import evaluate_clustering
from viz import project_2d, plot_clusters


def load_jsonl(path: str, text_field: str = "text"):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj[text_field])
            # если есть внешняя разметка (не обязательно)
            if "label" in obj:
                labels.append(obj["label"])
    y = np.array(labels) if labels else None
    return texts, y


def main():
    corpus_path = "data/corpus.jsonl"
    texts, y = load_jsonl(corpus_path)

    # 1) preprocessing + tokenization
    pre_cfg = PreprocessConfig(lower=True, lemmatize=True, min_token_len=2)
    tok_mode = "bpe"  # 'whitespace'|'regex'|'bpe'
    bpe_codes = "models/bpe.codes"

    tokens_list = preprocess_and_tokenize(texts, pre_cfg, tok_mode, bpe_codes_path=bpe_codes)

    # 2) vectorization
    vcfg = VectorConfig(method="tfidf", use_l2_norm=True)
    vec = TextVectorizer(vcfg).fit(tokens_list)
    X = vec.transform(tokens_list)

    # 3) clustering
    ccfg = ClusterConfig(method="kmeans", n_clusters=12)
    labels, meta = clusterize(X, ccfg)

    # 4) evaluation
    metrics = evaluate_clustering(X, labels, y_true=y)
    print("Metrics:", metrics)

    # 5) visualization
    Z2 = project_2d(X, method="umap")
    plot_clusters(Z2, labels, title=f"{vcfg.method}+{ccfg.method}")

    # save
    out = pd.DataFrame({"text": texts, "cluster": labels})
    out.to_csv("results_clusters.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
