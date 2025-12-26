from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re
import regex as regx

# Опционально: лемматизация
try:
    import pymorphy2
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

# BPE (из subword-nmt) — удобно, если в ЛР1 использовался именно он
try:
    from subword_nmt.apply_bpe import BPE
except Exception:
    BPE = None


@dataclass
class PreprocessConfig:
    lower: bool = True
    keep_only_letters: bool = False   # если True — выкинуть цифры/пунктуацию
    min_token_len: int = 2
    lemmatize: bool = True
    language: str = "ru"


class TextPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def clean(self, text: str) -> str:
        if self.cfg.lower:
            text = text.lower()
        text = text.replace("\u00ad", "")  # soft hyphen
        text = re.sub(r"\s+", " ", text).strip()

        if self.cfg.keep_only_letters:
            # оставить буквы + пробелы
            text = regx.sub(r"[^\p{L}\s]+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not self.cfg.lemmatize or _MORPH is None:
            return tokens
        lemmas = []
        for t in tokens:
            if len(t) < self.cfg.min_token_len:
                continue
            p = _MORPH.parse(t)[0]
            lemmas.append(p.normal_form)
        return lemmas


class Tokenizer:
    def __init__(self, mode: str, bpe_codes_path: Optional[str] = None):
        """
        mode: 'whitespace' | 'regex' | 'bpe'
        """
        self.mode = mode
        self._bpe = None

        if mode == "bpe":
            if BPE is None:
                raise ImportError("subword-nmt не установлен. Установи: pip install subword-nmt")
            if not bpe_codes_path:
                raise ValueError("Для BPE нужен путь к bpe.codes из ЛР1")
            with open(bpe_codes_path, "r", encoding="utf-8") as f:
                self._bpe = BPE(f)

        # regex токенизация: буквы/цифры внутри слов, дефисы как часть слова
        self._token_re = regx.compile(r"[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*", flags=regx.IGNORECASE)

    def tokenize(self, text: str) -> List[str]:
        if self.mode == "whitespace":
            return [t for t in text.split() if t]
        if self.mode == "regex":
            return self._token_re.findall(text)
        if self.mode == "bpe":
            # subword-nmt expects whitespace tokenized input; simplest: split then apply BPE on sentence string
            bpe_line = self._bpe.process_line(text)
            return [t for t in bpe_line.split() if t]
        raise ValueError(f"Unknown tokenizer mode: {self.mode}")


def preprocess_and_tokenize(
    texts: List[str],
    pre_cfg: PreprocessConfig,
    tok_mode: str,
    bpe_codes_path: Optional[str] = None
) -> List[List[str]]:
    prep = TextPreprocessor(pre_cfg)
    tok = Tokenizer(tok_mode, bpe_codes_path=bpe_codes_path)

    out = []
    for s in texts:
        s = prep.clean(s)
        tokens = tok.tokenize(s)
        tokens = [t for t in tokens if len(t) >= pre_cfg.min_token_len]
        tokens = prep.lemmatize_tokens(tokens)
        out.append(tokens)
    return out
