from huggingface_hub import HfApi, ModelCard
import os


class ModelPublisher:
    def __init__(self, username, token):
        self.api = HfApi()
        self.username = username
        self.token = token

    def create_model_card(self, model_name, metrics, corpus_info):
        """Создание карточки модели"""

        card_content = f"""
---
language:
- ru
license: mit
tags:
- russian
- tokenizer
- nlp
- BPE
---

# {model_name}

## 🗃️ Корпус
{corpus_info}

## ⚙️ Параметры
- Алгоритм: BPE
- Размер словаря: {metrics.get('vocab_size', 'N/A')}
- Min frequency: {metrics.get('min_frequency', 'N/A')}

## 📊 Метрики
- OOV rate: {metrics.get('oov_rate', 'N/A')}%
- Reconstruction accuracy: {metrics.get('reconstruction_accuracy', 'N/A')}%
- Compression ratio: {metrics.get('compression_ratio', 'N/A')}

## 💻 Пример использования

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{self.username}/{model_name}")
text = "Привет, как дела?"
tokens = tokenizer.tokenize(text)
print(tokens)"""