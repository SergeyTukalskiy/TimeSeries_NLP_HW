import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import json
from collections import Counter
import os
from typing import List, Dict, Optional


class RbcParser:
    def __init__(self):
        self.total_words = 0
        self.articles_collected = 0
        self.processed_urls = set()  # Для избежания дубликатов

    def _count_words(self, text):
        """Подсчитывает количество слов в тексте"""
        if not text:
            return 0
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def _get_url(self, param_dict: dict) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями
        """
        url = 'https://www.rbc.ru/search/ajax/?' + \
              'project={0}&'.format(param_dict['project']) + \
              'category={0}&'.format(param_dict['category']) + \
              'dateFrom={0}&'.format(param_dict['dateFrom']) + \
              'dateTo={0}&'.format(param_dict['dateTo']) + \
              'page={0}&'.format(param_dict['page']) + \
              'query={0}&'.format(param_dict['query']) + \
              'material={0}'.format(param_dict['material'])

        return url

    def _get_search_results(self, param_dict: dict) -> list:
        """
        Возвращает список статей с поисковой выдачи
        """
        url = self._get_url(param_dict)
        try:
            r = rq.get(url)
            r.raise_for_status()
            return r.json()['items']
        except Exception as e:
            print(f"Ошибка при получении поисковой выдачи: {e}")
            return []

    def _get_article_data(self, url: str):
        """
        Возвращает заголовок, текст статьи, категорию и дату по ссылке
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            r = rq.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = bs(r.text, features="lxml")

            # Получаем заголовок
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.text.strip() if title_elem else "Без заголовка"

            # Получаем дату публикации
            date = None
            date_elem = soup.find('time')
            if date_elem and date_elem.get('datetime'):
                date = date_elem.get('datetime')
            else:
                # Альтернативные способы найти дату
                date_span = soup.find('span', {'class': 'article__header__date'})
                if date_span:
                    date = date_span.text.strip()

            # Получаем основной текст
            text = ""
            article_body = soup.find('div', {'class': 'article__text'})
            if not article_body:
                article_body = soup.find('article')

            if article_body:
                paragraphs = article_body.find_all('p')
                text = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])

            # Получаем категорию/рубрику
            category = "Не указана"
            breadcrumbs = soup.find('div', {'class': 'article__header__breadcrumbs'})
            if breadcrumbs:
                category_links = breadcrumbs.find_all('a')
                if category_links:
                    category = category_links[-1].text.strip()

            if category == "Не указана":
                category_elem = soup.find('a', {'class': 'article__header__category'})
                if category_elem:
                    category = category_elem.text.strip()

            return title, text, category, date, url

        except Exception as e:
            print(f"Ошибка при парсинге статьи {url}: {e}")
            return None, None, None, None, None

    def _get_current_date_range(self):
        """Возвращает текущую дату и дату неделю назад для поиска"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        return start_date.strftime('%d.%m.%Y'), end_date.strftime('%d.%m.%Y')

    def collect_articles(self,
                         word_limit: int = 50000,
                         delay: float = 1.0) -> List[Dict]:
        """
        Собирает новости от самых свежих до достижения указанного лимита слов

        Args:
            word_limit: Целевое количество слов (по умолчанию 50000)
            delay: Задержка между запросами в секундах

        Returns:
            List[Dict] с собранными статьями
        """
        articles_data = []
        self.total_words = 0
        self.articles_collected = 0
        self.processed_urls.clear()

        # Начинаем с текущей даты и идем вглубь
        current_end_date = datetime.now()

        while self.total_words < word_limit:
            # Определяем диапазон дат для поиска (недельные интервалы)
            start_date = current_end_date - timedelta(days=7)

            start_str = start_date.strftime('%d.%m.%Y')
            end_str = current_end_date.strftime('%d.%m.%Y')

            print(f"Поиск за период: {start_str} - {end_str}")

            # Параметры для поиска
            params = {
                'project': 'rbcnews',
                'category': '',
                'dateFrom': start_str,
                'dateTo': end_str,
                'page': '1',
                'query': '',
                'material': 'news'
            }

            # Обрабатываем все страницы для текущего периода
            page = 1
            has_more_pages = True
            period_articles = 0

            while has_more_pages and self.total_words < word_limit:
                params['page'] = str(page)

                try:
                    search_results = self._get_search_results(params)

                    if not search_results:
                        has_more_pages = False
                        break

                    # Обрабатываем каждую статью на странице
                    for article in search_results:
                        if self.total_words >= word_limit:
                            break

                        url = article.get('fronturl')
                        if not url or url in self.processed_urls:
                            continue

                        # Получаем данные статьи
                        title, text, category, date, url = self._get_article_data(url)

                        if title and text and url:
                            word_count = self._count_words(text)

                            if word_count > 50:  # Игнорируем очень короткие статьи
                                article_dict = {
                                    'title': title,
                                    'text': text,
                                    'category': category,
                                    'date': date or article.get('publish_date', ''),
                                    'url': url,
                                    'source': 'rbc.ru'
                                }

                                articles_data.append(article_dict)
                                self.total_words += word_count
                                self.articles_collected += 1
                                period_articles += 1
                                self.processed_urls.add(url)

                                print(
                                    f"    Статья {self.articles_collected}: {word_count} слов | Всего: {self.total_words:,} слов")

                        # Задержка между запросами к статьям
                        time.sleep(delay)

                    # Проверяем, есть ли следующая страница
                    if len(search_results) < 20:  # Обычно на странице 20 статей
                        has_more_pages = False
                    else:
                        page += 1

                except Exception as e:
                    print(f"Ошибка при обработке страницы {page}: {e}")
                    has_more_pages = False

            print(f"  За период собрано: {period_articles} статей")

            # Переходим к предыдущей неделе
            current_end_date = start_date - timedelta(days=1)

            # Если за весь период не нашли статей, возможно, достигли предела архива
            if period_articles == 0:
                print("Возможно, достигнут предел архива новостей")
                break

        print("\n" + "=" * 50)
        print("СБОР ДАННЫХ ЗАВЕРШЕН")
        print(f"Всего собрано статей: {self.articles_collected}")
        print(f"Общее количество слов: {self.total_words:,}")
        print(f"Целевой лимит: {word_limit:,} слов")

        if self.total_words >= word_limit:
            print("✓ Лимит слов достигнут!")
        else:
            print("⚠ Лимит слов не достигнут (закончились статьи в архиве)")

        return articles_data


class NewsCorpusCollector:
    def __init__(self):
        self.articles = []
        self.rbc_parser = RbcParser()

    def collect_from_rbc(self, word_limit: int = 50000, delay: float = 0.5) -> List[Dict]:
        """Сбор статей с RBC"""
        print("🚀 Начинаем сбор статей с RBC...")
        rbc_articles = self.rbc_parser.collect_articles(word_limit=word_limit, delay=delay)
        self.articles.extend(rbc_articles)
        return rbc_articles

    def collect_from_static_source(self, url: str, source: str) -> Optional[Dict]:
        """Базовый метод для статических источников (можно расширить)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = rq.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = bs(response.content, 'html.parser')

            # Базовая структура - нужно адаптировать под каждый источник
            title = soup.find('h1')
            title = title.get_text().strip() if title else "Без заголовка"

            # Поиск основного текста (универсальные селекторы)
            text_elements = soup.find_all(['p', 'div'], class_=re.compile(r'article|content|text|body'))
            text = ' '.join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])

            if not text:
                # Альтернативный подход
                article = soup.find('article')
                if article:
                    text = article.get_text().strip()

            date = soup.find('time')
            date = date.get('datetime') if date and date.get('datetime') else None

            return {
                'title': title,
                'text': text,
                'date': date,
                'url': url,
                'source': source,
                'category': 'news'
            }
        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return None

    def collect_corpus(self, target_size: int = 50000, sources: List[str] = None):
        """
        Основной метод сбора корпуса

        Args:
            target_size: целевой размер корпуса в словах
            sources: список источников для парсинга
        """
        if sources is None:
            sources = ['rbc']  # По умолчанию только RBC

        current_words = self.get_total_words()

        for source in sources:
            if current_words >= target_size:
                break

            if source.lower() == 'rbc':
                print(f"\n📰 Сбор данных из источника: {source}")
                articles = self.collect_from_rbc(
                    word_limit=target_size - current_words,
                    delay=0.5  # Безопасная задержка
                )
                current_words = self.get_total_words()

            # Здесь можно добавить другие источники
            # elif source.lower() == 'ria':
            #     articles = self.collect_from_ria(...)

            print(f"✅ Собрано {len(articles)} статей из {source}")

    def get_total_words(self) -> int:
        """Подсчет общего количества слов во всех статьях"""
        total = 0
        for article in self.articles:
            if 'text' in article and article['text']:
                words = re.findall(r'\b\w+\b', article['text'])
                total += len(words)
        return total

    def get_statistics(self) -> Dict:
        """Получение статистики по корпусу"""
        if not self.articles:
            return {}

        total_articles = len(self.articles)
        total_words = self.get_total_words()

        # Статистика по источникам
        sources = Counter(article.get('source', 'unknown') for article in self.articles)

        # Статистика по категориям
        categories = Counter(article.get('category', 'Не указана') for article in self.articles)

        # Средняя длина статьи
        avg_words_per_article = total_words / total_articles if total_articles > 0 else 0

        return {
            'total_articles': total_articles,
            'total_words': total_words,
            'avg_words_per_article': avg_words_per_article,
            'sources': dict(sources),
            'categories': dict(categories)
        }

    def save_to_jsonl(self, filename: str = 'raw_corpus.jsonl'):
        """Сохранение корпуса в JSONL формат"""
        with open(filename, 'w', encoding='utf-8') as f:
            for article in self.articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"✅ Корпус сохранен в файл: {filename}")

        # Вывод статистики
        stats = self.get_statistics()
        if stats:
            print(f"\n📊 Статистика корпуса:")
            print(f"   Всего статей: {stats['total_articles']}")
            print(f"   Всего слов: {stats['total_words']:,}")
            print(f"   Среднее слов в статье: {stats['avg_words_per_article']:.1f}")
            print(f"   Источники: {stats['sources']}")

    def load_from_jsonl(self, filename: str) -> bool:
        """Загрузка корпуса из JSONL файла"""
        try:
            self.articles = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line.strip())
                    self.articles.append(article)

            print(f"✅ Корпус загружен из файла: {filename}")
            stats = self.get_statistics()
            print(f"📊 Загружено {stats['total_articles']} статей, {stats['total_words']:,} слов")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки корпуса: {e}")
            return False


# Функции для удобства использования
def create_sample_corpus(word_limit: int = 50000) -> NewsCorpusCollector:
    """Создание корпуса с настройками по умолчанию"""
    collector = NewsCorpusCollector()
    collector.collect_corpus(target_size=word_limit)
    return collector


def quick_collect(output_file: str = 'rbc_corpus.jsonl', word_limit: int = 10000):
    """Быстрый сбор небольшого корпуса для тестирования"""
    print("🚀 Быстрый сбор тестового корпуса...")
    collector = NewsCorpusCollector()
    collector.collect_from_rbc(word_limit=word_limit, delay=0.3)

    if collector.articles:
        collector.save_to_jsonl(output_file)
        return collector
    else:
        print("❌ Не удалось собрать статьи")
        return None


# Пример использования
if __name__ == "__main__":
    # Быстрый тест
    collector = quick_collect(word_limit=5000)

    # Полный сбор
    # collector = create_sample_corpus(word_limit=50000)
    # collector.save_to_jsonl('full_corpus.jsonl')