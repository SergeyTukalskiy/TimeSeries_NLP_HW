import argparse
import sys
import os

import nltk

from config import Config
from corpus_collector import quick_collect, NewsCorpusCollector


def main(arg):
    parser = argparse.ArgumentParser(description="NLP Tokenization Analysis Pipeline")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Парсинг корпуса
    parse_parser = subparsers.add_parser('parse', help='Parse news corpus')
    parse_parser.add_argument('--size', type=int, default=50000, help='Target corpus size in words')

    # Очистка данных
    clean_parser = subparsers.add_parser('clean', help='Clean and preprocess corpus')

    # Сравнение методов
    compare_parser = subparsers.add_parser('compare', help='Compare tokenization methods')
    compare_parser.add_argument('--sample', type=int, default=1000, help='Sample size for comparison')

    # Обучение моделей
    train_parser = subparsers.add_parser('train', help='Train subword models')
    train_parser.add_argument('--vocab-size', type=int, nargs='+', default=[8000, 16000, 32000])

    # Запуск веб-интерфейса
    web_parser = subparsers.add_parser('web', help='Start web interface')

    # Публикация моделей
    publish_parser = subparsers.add_parser('publish', help='Publish models to Hugging Face')
    publish_parser.add_argument('--model', required=True, help='Model name to publish')

    args = parser.parse_args()

    if arg == 'parse':
        run_parsing(50000)
    elif arg == 'clean':
        run_cleaning()
    elif arg == 'compare':
        run_comparison(1000)
    elif arg == 'train':
        run_training([2000, 3500, 5000])
    elif arg == 'web':
        run_web_interface()
    elif arg == 'publish':
        run_publishing('bpe_16000')
    else:
        parser.print_help()


def run_parsing(target_size):
    """Запуск парсинга корпуса"""
    print("🚀 Starting corpus parsing...")
    from corpus_collector import NewsCorpusCollector

    collector = NewsCorpusCollector()
    collector.collect_corpus(target_size=target_size)
    collector.save_to_jsonl(Config.RAW_CORPUS_PATH)

    print(f"✅ Corpus saved to {Config.RAW_CORPUS_PATH}")
    print(f"📊 Total words: {collector.get_total_words()}")


def run_cleaning():
    """Запуск очистки данных"""
    print("🧹 Starting data cleaning...")
    from text_cleaner import TextCleaner
    from universal_preprocessor import UniversalPreprocessor
    import json

    cleaner = TextCleaner()
    preprocessor = UniversalPreprocessor()

    cleaned_articles = []

    with open(Config.RAW_CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)

            # Очистка текста
            clean_text = cleaner.clean_pipeline(article['text'])

            # Препроцессинг
            processed_text = preprocessor.preprocess(clean_text)

            article['processed_text'] = processed_text
            cleaned_articles.append(article)

    # Сохранение очищенного корпуса
    with open(Config.CLEAN_CORPUS_PATH, 'w', encoding='utf-8') as f:
        for article in cleaned_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"✅ Cleaned corpus saved to {Config.CLEAN_CORPUS_PATH}")


def run_comparison(sample_size):
    """Запуск сравнения методов"""
    print("📊 Starting method comparison...")
    from tokenization_comparison import TokenizationComparator
    import json

    # Загрузка данных
    texts = []
    with open(Config.CLEAN_CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            texts.append(article['processed_text'])

    # Сравнение методов
    comparator = TokenizationComparator(texts[:sample_size])
    results = comparator.run_comparison(sample_size=min(sample_size, len(texts)))

    # Сохранение результатов
    import pandas as pd
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('results/tokenization_comparison.csv')

    print("✅ Comparison results saved to results/tokenization_comparison.csv")
    print(df)


def run_training(vocab_sizes):
    """Обучение подсловных моделей"""
    print("🤖 Training subword models...")
    from subword_training import SubwordTokenizerTrainer
    import json

    # Загрузка данных
    texts = []
    with open(Config.CLEAN_CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if 'processed_text' in article and article['processed_text']:
                texts.append(article['processed_text'])
            elif 'text' in article and article['text']:
                texts.append(article['text'])

    if not texts:
        print("❌ No texts found for training")
        return

    print(f"📝 Loaded {len(texts)} texts for training")

    trainer = SubwordTokenizerTrainer(texts)

    # Используем рекомендованные размеры если не указаны
    if not vocab_sizes:
        vocab_sizes = trainer.corpus_stats['recommended_sizes']
        print(f"🎯 Using recommended vocab sizes: {vocab_sizes}")

    for vocab_size in vocab_sizes:
        print(f"\n🔧 Training models with vocab_size={vocab_size}")

        try:
            # BPE
            print("  Training BPE...")
            bpe_tokenizer = trainer.train_bpe(vocab_size=vocab_size)
            bpe_tokenizer.save(f"models/bpe_{vocab_size}.json")
            print(f"  ✅ BPE saved: models/bpe_{vocab_size}.json")

            # WordPiece
            print("  Training WordPiece...")
            wp_tokenizer = trainer.train_wordpiece(vocab_size=vocab_size)
            wp_tokenizer.save(f"models/wordpiece_{vocab_size}.json")
            print(f"  ✅ WordPiece saved: models/wordpiece_{vocab_size}.json")

            # Unigram - теперь используем исправленный метод
            print("  Training Unigram...")
            unigram_result = trainer.train_unigram(vocab_size=vocab_size)
            # Unigram модели сохраняются автоматически при обучении
            print(f"  ✅ Unigram trained: vocab_size={unigram_result['vocab_size']}")

        except Exception as e:
            print(f"❌ Error training models with vocab_size={vocab_size}: {e}")
            continue

    print("✅ All models trained")


def run_web_interface():
    """Запуск веб-интерфейса"""
    print("🌐 Starting web interface...")
    print("Open http://localhost:8501 in your browser")

    # Запуск Streamlit
    os.system("streamlit run web_app.py")


def run_publishing(model_name):
    """Публикация моделей"""
    print(f"📤 Publishing model {model_name}...")
    from model_publisher import ModelPublisher

    publisher = ModelPublisher(
        username=Config.HF_USERNAME,
        token=Config.HF_TOKEN
    )

    # Здесь нужно добавить логику загрузки метрик модели
    metrics = {
        'vocab_size': 16000,
        'oov_rate': 1.2,
        'compression_ratio': 1.35
    }

    corpus_info = "50k+ words from rbc.ru"

    success = publisher.publish_model(
        model_path=f"models/{model_name}",
        model_name=model_name,
        metrics=metrics,
        corpus_info=corpus_info
    )

    if success:
        print(f"✅ Model {model_name} published successfully!")
    else:
        print("❌ Failed to publish model")


if __name__ == "__main__":
    main('web')