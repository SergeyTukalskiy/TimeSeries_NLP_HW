import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Any, Tuple
import scipy.sparse as sp
from collections import Counter
import time

class ClassicalVectorizers:
    """Класс для классических методов векторизации"""
    
    def __init__(self):
        self.vectorizers = {}
        self.vocabulary = None
        
    def prepare_texts(self, processed_corpus: List[Dict[str, Any]]) -> List[str]:
        """Подготовка текстов для векторизации"""
        return [' '.join(doc['processed_text']) for doc in processed_corpus]
    
    def one_hot_encoding(self, processed_corpus: List[Dict[str, Any]], ngram_range: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
        """One-Hot Encoding"""
        print("Применение One-Hot Encoding...")
        start_time = time.time()
        
        texts = self.prepare_texts(processed_corpus)
        
        # Создание словаря
        vectorizer = CountVectorizer(binary=True, ngram_range=ngram_range, max_features=10000)
        X = vectorizer.fit_transform(texts)
        
        self.vectorizers['one_hot'] = vectorizer
        self.vocabulary = vectorizer.get_feature_names_out()
        
        metrics = self._calculate_metrics(X, "One-Hot Encoding")
        metrics['time'] = time.time() - start_time
        
        return {
            'matrix': X,
            'vocabulary': self.vocabulary,
            'metrics': metrics
        }
    
    def bag_of_words(self, processed_corpus: List[Dict[str, Any]], 
                    binary: bool = False,
                    ngram_range: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
        """Bag of Words с различными схемами взвешивания"""
        print("Применение Bag of Words...")
        start_time = time.time()
        
        texts = self.prepare_texts(processed_corpus)
        
        vectorizer = CountVectorizer(binary=binary, ngram_range=ngram_range, max_features=10000)
        X = vectorizer.fit_transform(texts)
        
        method_name = "Binary BoW" if binary else "Frequency BoW"
        self.vectorizers['bow'] = vectorizer
        
        metrics = self._calculate_metrics(X, method_name)
        metrics['time'] = time.time() - start_time
        
        return {
            'matrix': X,
            'vocabulary': vectorizer.get_feature_names_out(),
            'metrics': metrics
        }
    
    def tfidf_vectorizer(self, processed_corpus: List[Dict[str, Any]],
                        ngram_range: Tuple[int, int] = (1, 1),
                        smooth_idf: bool = True,
                        sublinear_tf: bool = False) -> Dict[str, Any]:
        """TF-IDF векторизация"""
        print("Применение TF-IDF...")
        start_time = time.time()
        
        texts = self.prepare_texts(processed_corpus)
        
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            max_features=10000
        )
        X = vectorizer.fit_transform(texts)
        
        self.vectorizers['tfidf'] = vectorizer
        
        method_name = f"TF-IDF (smooth_idf={smooth_idf}, sublinear_tf={sublinear_tf})"
        metrics = self._calculate_metrics(X, method_name)
        metrics['time'] = time.time() - start_time
        
        return {
            'matrix': X,
            'vocabulary': vectorizer.get_feature_names_out(),
            'metrics': metrics
        }
    
    def ngram_analysis(self, processed_corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ n-грамм"""
        print("Анализ n-грамм...")
        
        results = {}
        
        # Uni-grams
        results['unigram'] = self.tfidf_vectorizer(processed_corpus, ngram_range=(1, 1))
        
        # Bi-grams
        results['bigram'] = self.tfidf_vectorizer(processed_corpus, ngram_range=(2, 2))
        
        # Tri-grams
        results['trigram'] = self.tfidf_vectorizer(processed_corpus, ngram_range=(3, 3))
        
        # Combined 1-3 grams
        results['combined'] = self.tfidf_vectorizer(processed_corpus, ngram_range=(1, 3))
        
        return results
    
    def _calculate_metrics(self, matrix, method_name: str) -> Dict[str, Any]:
        """Расчет метрик для матрицы"""
        n_docs, n_features = matrix.shape
        n_nonzero = matrix.nnz
        sparsity = 1.0 - (n_nonzero / (n_docs * n_features))
        
        return {
            'method': method_name,
            'dimensions': n_features,
            'sparsity': sparsity,
            'non_zero_elements': n_nonzero,
            'density': 1.0 - sparsity
        }
    
    def compare_methods(self, processed_corpus: List[Dict[str, Any]]) -> pd.DataFrame:
        """Сравнение всех методов векторизации"""
        print("Сравнение методов векторизации...")
        
        results = []
        
        # One-Hot Encoding
        one_hot_result = self.one_hot_encoding(processed_corpus)
        results.append(one_hot_result['metrics'])
        
        # Binary BoW
        bow_binary = self.bag_of_words(processed_corpus, binary=True)
        results.append(bow_binary['metrics'])
        
        # Frequency BoW
        bow_freq = self.bag_of_words(processed_corpus, binary=False)
        results.append(bow_freq['metrics'])
        
        # TF-IDF
        tfidf_result = self.tfidf_vectorizer(processed_corpus)
        results.append(tfidf_result['metrics'])
        
        # N-gram analysis
        ngram_results = self.ngram_analysis(processed_corpus)
        for key, result in ngram_results.items():
            results.append(result['metrics'])
        
        return pd.DataFrame(results)