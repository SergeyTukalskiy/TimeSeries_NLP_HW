import re
from typing import Iterable, List


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^а-яА-Яa-zA-Z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_corpus(docs: Iterable[dict]) -> List[str]:
    """
    Объединяем title + text и чистим.
    Возвращаем список строк (готовый вход в TF-IDF).
    """
    processed = []
    for d in docs:
        txt = f"{d.get('title', '')}. {d.get('text', '')}"
        processed.append(basic_clean(txt))
    return processed
