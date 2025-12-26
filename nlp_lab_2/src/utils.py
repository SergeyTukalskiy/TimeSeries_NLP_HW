import json
import re
import pandas as pd
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
from tqdm import tqdm

try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except:
    pass

class TextPreprocessor:
    """Класс для предварительной обработки текстов"""
    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        # Дополнительные стоп-слова для новостных текстов
        self.stop_words.update(['это', 'тот', 'который', 'весь', 'такой', 'свой'])
        
    def clean_text(self, text: str) -> str:
        """Очистка текста от шума"""
        if not isinstance(text, str):
            return ""
        
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', '', text)
        # Удаление URL
        text = re.sub(r'http\S+', '', text)
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        # Удаление специальных символов, оставляя буквы, цифры и основные знаки препинания
        text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s\.\,\!\?\-\:]', ' ', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        # Приведение к нижнему регистру
        text = text.lower().strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        return word_tokenize(text, language='russian')
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Лемматизация токенов"""
        lemmas = []
        for token in tokens:
            if token.isalpha() and len(token) > 2 and token not in self.stop_words:
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                lemmas.append(lemma)
        return lemmas
    
    def preprocess_text(self, text: str) -> List[str]:
        """Полная предобработка текста"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        lemmas = self.lemmatize(tokens)
        return lemmas

class CorpusLoader:
    """Класс для загрузки и обработки корпуса"""
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        
    def load_corpus(self, filepath: str) -> List[Dict[str, Any]]:
        """Загрузка корпуса из JSONL файла"""
        corpus = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Загрузка корпуса"):
                try:
                    item = json.loads(line.strip())
                    corpus.append(item)
                except json.JSONDecodeError:
                    continue
        return corpus
    
    def process_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обработка всего корпуса"""
        processed_corpus = []
        
        for doc in tqdm(corpus, desc="Обработка документов"):
            processed_doc = doc.copy()
            text = doc.get('text', '')
            
            # Предобработка текста
            processed_text = self.preprocessor.preprocess_text(text)
            processed_doc['processed_text'] = processed_text
            processed_doc['word_count'] = len(processed_text)
            
            processed_corpus.append(processed_doc)
            
        return processed_corpus
    
    def save_processed_corpus(self, processed_corpus: List[Dict[str, Any]], filepath: str):
        """Сохранение обработанного корпуса"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in processed_corpus:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    def get_vocabulary(self, processed_corpus: List[Dict[str, Any]]) -> List[str]:
        """Получение словаря корпуса"""
        vocabulary = set()
        for doc in processed_corpus:
            vocabulary.update(doc['processed_text'])
        return sorted(list(vocabulary))