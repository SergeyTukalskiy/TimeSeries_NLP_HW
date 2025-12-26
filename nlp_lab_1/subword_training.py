import json
import re
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Sequence
import sentencepiece as spm
import os
import tempfile
from typing import Dict, List, Any
import logging
import shutil

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubwordTokenizerTrainer:
    def __init__(self, corpus_texts: List[str]):
        self.corpus_texts = [text for text in corpus_texts if text and isinstance(text, str)]
        self.corpus_file = "corpus.txt"

        # Сохраняем корпус в файл
        self._prepare_corpus()

        # Анализируем корпус для определения оптимальных параметров
        self.corpus_stats = self._analyze_corpus()

    def _prepare_corpus(self):
        """Подготовка корпуса для обучения"""
        logger.info(f"📝 Подготовка корпуса из {len(self.corpus_texts)} текстов...")

        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            for text in self.corpus_texts:
                # Базовая очистка текста
                cleaned_text = re.sub(r'\s+', ' ', text.strip())
                if cleaned_text:
                    f.write(cleaned_text + '\n')

        logger.info(f"✅ Корпус сохранен в {self.corpus_file}")

    def _analyze_corpus(self) -> Dict[str, Any]:
        """Анализ корпуса для определения оптимальных параметров"""
        logger.info("📊 Анализ корпуса...")

        all_text = ' '.join(self.corpus_texts)
        words = re.findall(r'\b\w+\b', all_text)
        unique_words = set(words)

        # Подсчет статистики
        total_words = len(words)
        total_chars = len(all_text)
        vocab_size_estimate = len(unique_words)

        logger.info(f"   Всего слов: {total_words:,}")
        logger.info(f"   Уникальных слов: {vocab_size_estimate:,}")
        logger.info(f"   Общее количество символов: {total_chars:,}")

        # Определяем максимальный размер словаря (ограничение для Unigram)
        max_unigram_vocab = min(7000, vocab_size_estimate + 1000)

        # Рекомендуемые размеры словаря
        recommended_sizes = [
            min(2000, max_unigram_vocab),
            min(5000, max_unigram_vocab),
            min(8000, max_unigram_vocab)
        ]

        # Убираем дубликаты и сортируем
        recommended_sizes = sorted(set(recommended_sizes))

        return {
            'total_words': total_words,
            'unique_words': vocab_size_estimate,
            'total_chars': total_chars,
            'max_unigram_vocab': max_unigram_vocab,
            'recommended_sizes': recommended_sizes
        }

    def train_bpe(self, vocab_size: int = 5000) -> Tokenizer:
        """Обучение BPE токенизатора"""
        logger.info(f"🔤 Обучение BPE модели с vocab_size={vocab_size}")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            show_progress=True
        )

        try:
            tokenizer.train([self.corpus_file], trainer)
            logger.info(f"✅ BPE модель обучена, словарь: {tokenizer.get_vocab_size()} токенов")
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Ошибка обучения BPE: {e}")
            raise

    def train_wordpiece(self, vocab_size: int = 5000) -> Tokenizer:
        """Обучение WordPiece токенизатора"""
        logger.info(f"🔤 Обучение WordPiece модели с vocab_size={vocab_size}")

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            show_progress=True
        )

        try:
            tokenizer.train([self.corpus_file], trainer)
            logger.info(f"✅ WordPiece модель обучена, словарь: {tokenizer.get_vocab_size()} токенов")
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Ошибка обучения WordPiece: {e}")
            raise

    def train_unigram(self, vocab_size: int = 5000) -> Dict[str, Any]:
        """Обучение Unigram модели с помощью SentencePiece"""
        logger.info(f"🔤 Обучение Unigram модели с vocab_size={vocab_size}")

        # Проверяем, что vocab_size не превышает максимальный
        if vocab_size > self.corpus_stats['max_unigram_vocab']:
            logger.warning(
                f"⚠ Слишком большой vocab_size для Unigram. Уменьшаем до {self.corpus_stats['max_unigram_vocab']}")
            vocab_size = self.corpus_stats['max_unigram_vocab']

        model_prefix = f"unigram_model_{vocab_size}"

        try:
            # Параметры для SentencePiece
            spm.SentencePieceTrainer.train(
                input=self.corpus_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=0.9995,
                model_type='unigram',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece='[PAD]',
                unk_piece='[UNK]',
                bos_piece='[BOS]',
                eos_piece='[EOS]',
                num_threads=4
            )

            # Загрузка обученной модели
            sp = spm.SentencePieceProcessor()
            sp.load(f'{model_prefix}.model')

            logger.info(f"✅ Unigram модель обучена, словарь: {sp.get_piece_size()} токенов")

            # Возвращаем и модель, и информацию о файлах для сохранения
            return {
                'model': sp,
                'model_files': {
                    'model_file': f'{model_prefix}.model',
                    'vocab_file': f'{model_prefix}.vocab'
                },
                'vocab_size': vocab_size
            }

        except Exception as e:
            logger.error(f"❌ Ошибка обучения Unigram: {e}")
            # Пробуем с меньшим vocab_size
            if vocab_size > 3000:
                logger.info("🔄 Пробуем обучить с vocab_size=3000...")
                return self.train_unigram(3000)
            else:
                raise

    def train_all_models(self, vocab_sizes: List[int] = None) -> Dict[str, Any]:
        """Обучение всех моделей с разными размерами словаря"""
        if vocab_sizes is None:
            vocab_sizes = self.corpus_stats['recommended_sizes']

        logger.info(f"🎯 Обучение моделей с размерами словаря: {vocab_sizes}")

        results = {
            'bpe': {},
            'wordpiece': {},
            'unigram': {},
            'corpus_stats': self.corpus_stats
        }

        for vocab_size in vocab_sizes:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"📏 Размер словаря: {vocab_size}")
            logger.info(f"{'=' * 50}")

            try:
                # BPE
                bpe_model = self.train_bpe(vocab_size)
                results['bpe'][vocab_size] = {
                    'model': bpe_model,
                    'vocab_size': vocab_size
                }

                # WordPiece
                wp_model = self.train_wordpiece(vocab_size)
                results['wordpiece'][vocab_size] = {
                    'model': wp_model,
                    'vocab_size': vocab_size
                }

                # Unigram
                unigram_result = self.train_unigram(vocab_size)
                results['unigram'][vocab_size] = unigram_result

            except Exception as e:
                logger.error(f"❌ Ошибка при обучении с vocab_size={vocab_size}: {e}")
                continue

        return results

    def evaluate_tokenizer(self, tokenizer, tokenizer_type: str, test_texts: List[str]) -> Dict[str, float]:
        """Оценка токенизатора на тестовых текстах"""
        logger.info(f"📊 Оценка {tokenizer_type} токенизатора...")

        fragmentation_rates = []
        compression_ratios = []
        token_counts = []

        for text in test_texts[:100]:  # Оцениваем на 100 текстах
            if not text:
                continue

            original_words = len(re.findall(r'\b\w+\b', text))

            if tokenizer_type == 'unigram':
                # Для SentencePiece
                tokens = tokenizer.encode_as_pieces(text)
            else:
                # Для tokenizers
                tokens = tokenizer.encode(text).tokens

            token_count = len(tokens)

            # Процент фрагментации (слова, разбитые на подслова)
            fragmented_words = sum(1 for token in tokens if '##' in token or '▁' in token)
            fragmentation_rate = fragmented_words / token_count if token_count > 0 else 0

            # Коэффициент сжатия
            compression_ratio = original_words / token_count if token_count > 0 else 1

            fragmentation_rates.append(fragmentation_rate)
            compression_ratios.append(compression_ratio)
            token_counts.append(token_count)

        return {
            'avg_fragmentation_rate': sum(fragmentation_rates) / len(fragmentation_rates) if fragmentation_rates else 0,
            'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1,
            'avg_tokens_per_text': sum(token_counts) / len(token_counts) if token_counts else 0,
            'total_tokens_evaluated': sum(token_counts)
        }

    def save_models(self, models_dict: Dict[str, Any], output_dir: str = "models"):
        """Сохранение обученных моделей"""
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"💾 Сохранение моделей в {output_dir}...")

        saved_models = {}

        for model_type, vocab_models in models_dict.items():
            if model_type == 'corpus_stats':
                continue

            saved_models[model_type] = {}

            for vocab_size, model_data in vocab_models.items():
                if model_type in ['bpe', 'wordpiece']:
                    # Сохранение tokenizers моделей
                    filename = f"{output_dir}/{model_type}_{vocab_size}.json"
                    model_data['model'].save(filename)
                    saved_models[model_type][vocab_size] = {
                        'file': filename,
                        'vocab_size': vocab_size
                    }
                    logger.info(f"✅ {model_type}_{vocab_size} сохранен")

                elif model_type == 'unigram':
                    # Копируем файлы SentencePiece
                    model_files = model_data['model_files']
                    for file_type, src_file in model_files.items():
                        dst_file = f"{output_dir}/{model_type}_{vocab_size}.{file_type.split('_')[0]}"
                        if os.path.exists(src_file):
                            shutil.copy2(src_file, dst_file)
                            logger.info(f"✅ {dst_file} скопирован")

                    saved_models[model_type][vocab_size] = {
                        'model_file': f"{output_dir}/{model_type}_{vocab_size}.model",
                        'vocab_file': f"{output_dir}/{model_type}_{vocab_size}.vocab",
                        'vocab_size': vocab_size
                    }

        # Сохранение статистики корпуса
        stats_file = f"{output_dir}/corpus_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(models_dict['corpus_stats'], f, ensure_ascii=False, indent=2)

        # Сохранение информации о моделях
        models_info_file = f"{output_dir}/models_info.json"
        with open(models_info_file, 'w', encoding='utf-8') as f:
            json.dump(saved_models, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Все модели сохранены в {output_dir}")
        return saved_models

    def load_model(self, model_type: str, vocab_size: int, models_dir: str = "models"):
        """Загрузка сохраненной модели"""
        try:
            if model_type in ['bpe', 'wordpiece']:
                # Загрузка tokenizers моделей
                model_path = f"{models_dir}/{model_type}_{vocab_size}.json"
                tokenizer = Tokenizer.from_file(model_path)
                return tokenizer

            elif model_type == 'unigram':
                # Загрузка SentencePiece модели
                model_path = f"{models_dir}/{model_type}_{vocab_size}.model"
                sp = spm.SentencePieceProcessor()
                sp.load(model_path)
                return sp

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {model_type}_{vocab_size}: {e}")
            return None


def load_corpus_from_jsonl(filename: str) -> List[str]:
    """Загрузка корпуса из JSONL файла"""
    import json
    corpus = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    if 'text' in article and article['text']:
                        corpus.append(article['text'])
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка JSON в строке: {e}")
                    continue

        logger.info(f"📁 Загружено {len(corpus)} текстов из {filename}")
        return corpus

    except FileNotFoundError:
        logger.error(f"❌ Файл {filename} не найден")
        return []


def quick_train_test():
    """Быстрое тестирование обучения на маленьком корпусе"""
    logger.info("🧪 Быстрый тест обучения моделей...")

    # Тестовые данные
    test_texts = [
                     "Это пример текста для тестирования токенизации.",
                     "Здесь несколько предложений на русском языке.",
                     "Токенизация разбивает текст на отдельные слова.",
                     "Мы тестируем различные методы обработки текста.",
                     "Подсловные методы помогают обрабатывать редкие слова."
                 ] * 20  # Умножаем для большего объема

    logger.info(f"📝 Тестовый корпус: {len(test_texts)} текстов")

    try:
        trainer = SubwordTokenizerTrainer(test_texts)

        # Обучаем с маленьким словарем
        results = trainer.train_all_models(vocab_sizes=[1000, 2000])

        # Сохраняем модели
        saved_models = trainer.save_models(results, "test_models")

        # Тестируем на примере
        test_text = "Это пример текста для тестирования токенизации подсловными методами."

        for model_type, vocab_models in saved_models.items():
            for vocab_size, model_info in vocab_models.items():
                logger.info(f"\n🔍 Тест {model_type}_{vocab_size}:")

                # Загружаем модель для тестирования
                model = trainer.load_model(model_type, vocab_size, "test_models")
                if model:
                    if model_type == 'unigram':
                        tokens = model.encode_as_pieces(test_text)
                    else:
                        tokens = model.encode(test_text).tokens

                    logger.info(f"   Текст: {test_text}")
                    logger.info(f"   Токены: {tokens}")
                    logger.info(f"   Количество токенов: {len(tokens)}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в быстром тесте: {e}")
        return False


if __name__ == "__main__":
    # Быстрый тест
    if quick_train_test():
        logger.info("✅ Быстрый тест завершен успешно!")
    else:
        logger.error("❌ Быстрый тест завершен с ошибками")