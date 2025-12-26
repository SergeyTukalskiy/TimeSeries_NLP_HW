import re
import html

from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class TextCleaner:
    def __init__(self, language='russian'):
        self.stop_words = set(stopwords.words(language))

    def clean_html(self, text):
        """Удаление HTML разметки"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def remove_special_chars(self, text):
        """Удаление специальных символов"""
        # Сохраняем кириллицу, базовую пунктуацию
        text = re.sub(r'[^а-яёА-ЯЁ0-9\s\.\,\!\?\-\:\(\)]', ' ', text)
        return text

    def normalize_whitespace(self, text):
        """Стандартизация пробелов"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_stopwords(self, tokens):
        """Фильтрация стоп-слов"""
        return [token for token in tokens if token not in self.stop_words]

    def clean_pipeline(self, text, remove_stops=True, lowercase=True):
        """Полный пайплайн очистки"""
        # Декодирование HTML entities
        text = html.unescape(text)

        # Очистка HTML
        text = self.clean_html(text)

        # Приведение к нижнему регистру (опционально)
        if lowercase:
            text = text.lower()

        # Удаление специальных символов
        text = self.remove_special_chars(text)

        # Нормализация пробелов
        text = self.normalize_whitespace(text)

        # Токенизация для удаления стоп-слов
        if remove_stops:
            tokens = text.split()
            tokens = self.remove_stopwords(tokens)
            text = ' '.join(tokens)

        return text