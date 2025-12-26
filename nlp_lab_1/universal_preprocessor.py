import re


class UniversalPreprocessor:
    def __init__(self):
        self.patterns = {
            'url': r'https?://\S+|www\.\S+',
            'email': r'\S+@\S+',
            'number': r'\b\d+[\.,]?\d*\b',
            'abbreviations': {
                r'\bт\.\s*е\.\b': 'то есть',
                r'\bт\.\s*д\.\b': 'так далее',
                r'\bг\.\b': 'год',
                # Добавьте другие сокращения
            }
        }

    def standardize_punctuation(self, text):
        """Стандартизация пунктуации"""
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)
        text = re.sub(r'—', '-', text)
        return text

    def replace_special_tokens(self, text):
        """Замена специальных токенов"""
        # URL
        text = re.sub(self.patterns['url'], '<URL>', text)

        # Email
        text = re.sub(self.patterns['email'], '<EMAIL>', text)

        # Числа
        text = re.sub(self.patterns['number'], '<NUM>', text)

        return text

    def expand_abbreviations(self, text):
        """Раскрытие сокращений"""
        for abbrev, expansion in self.patterns['abbreviations'].items():
            text = re.sub(abbrev, expansion, text)
        return text

    def preprocess(self, text):
        """Основной пайплайн препроцессинга"""
        text = self.standardize_punctuation(text)
        text = self.expand_abbreviations(text)
        text = self.replace_special_tokens(text)
        return text