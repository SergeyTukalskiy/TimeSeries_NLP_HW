import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import TextPreprocessor, CorpusLoader
from classical_vectorizers import ClassicalVectorizers
from dimensionality_reduction import DimensionalityReducer
from distributed_models import DistributedModels
from semantic_analysis import SemanticAnalyzer

def main():
    """Основная функция для запуска экспериментов"""
    print("🚀 Запуск экспериментов по векторизации текстов")
    
    # Инициализация компонентов
    preprocessor = TextPreprocessor()
    corpus_loader = CorpusLoader(preprocessor)
    
    # 1. Загрузка и обработка корпуса
    print("\n📁 Этап 1: Загрузка и обработка корпуса")
    corpus = corpus_loader.load_corpus('data/rbc_articles_words.jsonl')
    processed_corpus = corpus_loader.process_corpus(corpus)
    
    # Сохранение обработанного корпуса
    corpus_loader.save_processed_corpus(processed_corpus, 'data/processed_corpus.jsonl')
    
    # Проверка объема корпуса
    total_words = sum(doc['word_count'] for doc in processed_corpus)
    print(f"📊 Общий объем корпуса: {total_words} слов")
    print(f"📄 Количество документов: {len(processed_corpus)}")
    
    # 2. Классические методы векторизации
    print("\n🔢 Этап 2: Классические методы векторизации")
    vectorizers = ClassicalVectorizers()
    
    # Сравнение методов
    comparison_df = vectorizers.compare_methods(processed_corpus)
    comparison_df.to_csv('vectorization_metrics.csv', index=False, encoding='utf-8')
    print("✅ Метрики векторизации сохранены в vectorization_metrics.csv")
    
    # 3. Снижение размерности
    print("\n📉 Этап 3: Снижение размерности")
    reducer = DimensionalityReducer()
    
    # Получаем TF-IDF матрицу для снижения размерности
    tfidf_result = vectorizers.tfidf_vectorizer(processed_corpus)
    tfidf_matrix = tfidf_result['matrix']
    
    # Поиск оптимального числа компонент
    optimal_components = reducer.find_optimal_components(tfidf_matrix)
    print(f"🎯 Оптимальное число компонент: {optimal_components['optimal_components']}")
    
    # Применение SVD
    svd_result = reducer.apply_svd(tfidf_matrix, n_components=100)
    print(f"📊 Объясненная дисперсия: {svd_result['explained_variance']:.4f}")
    
    # 4. Модели распределенных представлений
    print("\n🧠 Этап 4: Обучение моделей распределенных представлений")
    
    # Подготовка данных
    sentences = [doc['processed_text'] for doc in processed_corpus]
    
    # Обучение моделей
    dist_models = DistributedModels(vector_size=100, window=5, min_count=5)
    
    # Word2Vec
    w2v_skipgram = dist_models.train_word2vec(sentences, sg=1)
    w2v_cbow = dist_models.train_word2vec(sentences, sg=0)
    
    # FastText (используем gensim)
    ft_skipgram = dist_models.train_fasttext(sentences, sg=1)
    ft_cbow = dist_models.train_fasttext(sentences, sg=0)
    
    # Сравнение моделей
    models_comparison = dist_models.compare_models(sentences, processed_corpus)
    models_comparison.to_csv('models_comparison.csv', index=False, encoding='utf-8')
    print("✅ Сравнение моделей сохранено в models_comparison.csv")
    
    # 5. Семантический анализ
    print("\n🔍 Этап 5: Семантический анализ")
    semantic_analyzer = SemanticAnalyzer()
    
    # Тестирование векторной арифметики
    test_analogies = [
        (['москва', 'франция'], ['россия'], 'париж'),
        (['король', 'женщина'], ['мужчина'], 'королева')
    ]
    
    print("🧪 Тестирование векторной арифметики:")
    for pos, neg, expected in test_analogies:
        result = semantic_analyzer.vector_arithmetic(w2v_skipgram, pos, neg, topn=3)
        if result:
            print(f"  {pos} - {neg} = {result[0][0]} (ожидалось: {expected})")
    
    # Сохранение моделей
    print("\n💾 Сохранение моделей...")
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    w2v_skipgram.save(os.path.join(models_dir, 'word2vec_skipgram.model'))
    w2v_cbow.save(os.path.join(models_dir, 'word2vec_cbow.model'))
    ft_skipgram.save(os.path.join(models_dir, 'fasttext_skipgram.model'))
    ft_cbow.save(os.path.join(models_dir, 'fasttext_cbow.model'))
    
    print("✅ Все эксперименты завершены!")
    
    return {
        'processed_corpus': processed_corpus,
        'vectorizers': vectorizers,
        'reducer': reducer,
        'distributed_models': dist_models,
        'semantic_analyzer': semantic_analyzer
    }

if __name__ == "__main__":
    results = main()