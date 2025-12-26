import time
from collections import Counter
import spacy
import nltk
from razdel import tokenize as razdel_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TokenizationComparator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.nlp = spacy.load("ru_core_news_sm")
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")

    def naive_tokenization(self, text):
        """Наивная токенизация по пробелам"""
        return text.split()

    def razdel_tokenization(self, text):
        """Токенизация с помощью razdel"""
        return [token.text for token in razdel_tokenize(text)]

    def spacy_tokenization(self, text):
        """Токенизация с помощью spaCy"""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def spacy_lemmatization(self, text):
        """Лемматизация с помощью spaCy"""
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def pymorphy_lemmatization(self, text):
        """Лемматизация с помощью pymorphy2"""
        tokens = self.razdel_tokenization(text)
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def snowball_stemming(self, text):
        """Стемминг Snowball"""
        tokens = self.razdel_tokenization(text)
        return [self.stemmer.stem(token) for token in tokens]

    def evaluate_method(self, method_func, sample_texts):
        """Оценка метода по различным метрикам"""
        start_time = time.time()

        all_tokens = []
        oov_count = 0
        total_tokens = 0

        for text in sample_texts:
            tokens = method_func(text)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)

        vocab_size = len(set(all_tokens))
        processing_time = time.time() - start_time

        # Расчет OOV rate (упрощенный)
        word_freq = Counter(all_tokens)
        rare_words = sum(1 for word, freq in word_freq.items() if freq == 1)
        oov_rate = rare_words / len(all_tokens) if all_tokens else 0

        return {
            'vocab_size': vocab_size,
            'oov_rate': oov_rate,
            'processing_time': processing_time,
            'total_tokens': total_tokens,
            'avg_tokens_per_doc': total_tokens / len(sample_texts)
        }

    def run_comparison(self, sample_size=1000):
        """Запуск сравнения всех методов"""
        sample_texts = self.corpus[:sample_size]

        methods = {
            'Naive': self.naive_tokenization,
            'Razdel': self.razdel_tokenization,
            'SpaCy': self.spacy_tokenization,
            'SpaCy Lemma': self.spacy_lemmatization,
            'Pymorphy Lemma': self.pymorphy_lemmatization,
            'Snowball Stem': self.snowball_stemming
        }

        results = {}
        for name, method in methods.items():
            print(f"Testing {name}...")
            results[name] = self.evaluate_method(method, sample_texts)

        return results