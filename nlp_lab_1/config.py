import os


class Config:
    # Пути к данным
    DATA_DIR = "data"
    RAW_CORPUS_PATH = os.path.join(DATA_DIR, "raw_corpus.jsonl")
    CLEAN_CORPUS_PATH = os.path.join(DATA_DIR, "clean_corpus.jsonl")

    # Настройки парсинга
    NEWS_SOURCES = [
        "https://rbc.ru/",
        # Добавьте другие источники
    ]

    # Настройки токенизации
    VOCAB_SIZES = [8000, 16000, 32000]
    SAMPLE_SIZE = 1000  # для экспериментов

    # Hugging Face
    HF_USERNAME = "your_username"
    HF_TOKEN = "your_token"

    # Создание папок при инициализации
    @staticmethod
    def setup_directories():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)


# Инициализация при импорте
Config.setup_directories()