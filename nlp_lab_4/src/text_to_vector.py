from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rank_bm25 import BM25Okapi

from gensim.models import Word2Vec, FastText, KeyedVectors


@dataclass
class VectorConfig:
    method: str  # 'tfidf' | 'bm25' | 'w2v' | 'ft' | 'glove'
    max_features: int = 50000
    min_df: int = 3
    max_df: float = 0.95
    use_l2_norm: bool = True
    embedding_dim: int = 300


def _tokens_to_text(tokens_list: List[List[str]]) -> List[str]:
    return [" ".join(toks) for toks in tokens_list]


class TextVectorizer:
    def __init__(self, cfg: VectorConfig):
        self.cfg = cfg
        self.tfidf: Optional[TfidfVectorizer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.vocab: Optional[Dict[str, int]] = None

        self.emb: Optional[KeyedVectors] = None  # unified interface

    def fit(self, tokens_list: List[List[str]]):
        if self.cfg.method == "tfidf":
            self.tfidf = TfidfVectorizer(
                max_features=self.cfg.max_features,
                min_df=self.cfg.min_df,
                max_df=self.cfg.max_df,
                token_pattern=r"(?u)\b\w+\b"
            )
            self.tfidf.fit(_tokens_to_text(tokens_list))

        elif self.cfg.method == "bm25":
            self.bm25 = BM25Okapi(tokens_list)
            # для совместимости можно собрать vocab
            vocab = {}
            for doc in tokens_list:
                for t in doc:
                    if t not in vocab:
                        vocab[t] = len(vocab)
            self.vocab = vocab

        else:
            # эмбеддинги не "fit" на этом этапе, мы их грузим заранее
            pass

        return self

    def transform(self, tokens_list: List[List[str]]) -> np.ndarray:
        if self.cfg.method == "tfidf":
            X = self.tfidf.transform(_tokens_to_text(tokens_list)).astype(np.float32)
            # TF-IDF обычно sparse; для некоторых методов нужно dense
            X = X.toarray()

        elif self.cfg.method == "bm25":
            # BM25Okapi не даёт "вектор документа" напрямую как матрицу,
            # но можно построить bm25-матрицу (doc x term) по vocab.
            if self.bm25 is None or self.vocab is None:
                raise RuntimeError("BM25 vectorizer not fitted.")
            X = np.zeros((len(tokens_list), len(self.vocab)), dtype=np.float32)
            for i, doc in enumerate(tokens_list):
                # scores for terms in doc относительно всех термов не даются напрямую,
                # поэтому считаем bm25 веса только для термов документа
                # и кладём их в соответствующие позиции.
                for t in set(doc):
                    if t in self.vocab:
                        j = self.vocab[t]
                        # BM25Okapi.get_scores ожидает query, возвращает score по документам,
                        # поэтому здесь проще использовать internal idf + tf:
                        # Но для лабораторной достаточно такого “псевдо-вектора”:
                        # вес = idf(t) * tf(t,doc)
                        tf = doc.count(t)
                        idf = self.bm25.idf.get(t, 0.0)
                        X[i, j] = float(idf) * float(tf)

        else:
            # эмбеддинги: среднее по токенам
            if self.emb is None:
                raise RuntimeError("Embedding model not loaded (self.emb is None).")
            dim = self.emb.vector_size
            X = np.zeros((len(tokens_list), dim), dtype=np.float32)
            for i, doc in enumerate(tokens_list):
                vecs = []
                for t in doc:
                    if t in self.emb:
                        vecs.append(self.emb[t])
                if vecs:
                    X[i] = np.mean(vecs, axis=0)
                else:
                    X[i] = 0.0

        if self.cfg.use_l2_norm:
            X = normalize(X, norm="l2")
        return X

    def load_embeddings(self, path: str, kind: str):
        """
        kind: 'w2v' | 'ft' | 'glove'
        """
        if kind == "w2v":
            m = Word2Vec.load(path)
            self.emb = m.wv
        elif kind == "ft":
            m = FastText.load(path)
            self.emb = m.wv
        elif kind == "glove":
            # рекомендовано заранее конвертнуть в KeyedVectors и сохранить .kv
            self.emb = KeyedVectors.load(path)
        else:
            raise ValueError(f"Unknown embedding kind: {kind}")
        return self
