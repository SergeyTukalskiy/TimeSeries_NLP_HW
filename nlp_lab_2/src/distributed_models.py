import gensim
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import time
import os
from collections import defaultdict

class DistributedModels:
    """Класс для обучения моделей распределенных представлений"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.models = {}
        
    def train_word2vec(self, sentences: List[List[str]], sg: int = 1, 
                      epochs: int = 10, **kwargs) -> Word2Vec:
        """Обучение Word2Vec модели"""
        print(f"Обучение Word2Vec (sg={sg})...")
        start_time = time.time()
        
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=sg,
            epochs=epochs,
            **kwargs
        )
        
        training_time = time.time() - start_time
        model_name = f"word2vec_skipgram" if sg == 1 else f"word2vec_cbow"
        self.models[model_name] = {
            'model': model,
            'training_time': training_time
        }
        
        return model
    
    def train_fasttext(self, sentences: List[List[str]], sg: int = 1,
                      epochs: int = 10, **kwargs) -> FastText:
        """Обучение FastText модели с использованием gensim"""
        print(f"Обучение FastText (sg={sg})...")
        start_time = time.time()
        
        model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=sg,
            epochs=epochs,
            **kwargs
        )
        
        training_time = time.time() - start_time
        model_name = f"fasttext_skipgram" if sg == 1 else f"fasttext_cbow"
        self.models[model_name] = {
            'model': model,
            'training_time': training_time
        }
        
        return model
    
    def train_doc2vec(self, documents: List[TaggedDocument], dm: int = 1,
                     epochs: int = 10, **kwargs) -> Doc2Vec:
        """Обучение Doc2Vec модели"""
        print(f"Обучение Doc2Vec (dm={dm})...")
        start_time = time.time()
        
        model = Doc2Vec(
            documents=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            dm=dm,
            epochs=epochs,
            **kwargs
        )
        
        training_time = time.time() - start_time
        model_name = f"doc2vec_pvdm" if dm == 1 else f"doc2vec_pvdbow"
        self.models[model_name] = {
            'model': model,
            'training_time': training_time
        }
        
        return model
    
    def prepare_doc2vec_data(self, processed_corpus: List[Dict[str, Any]]) -> List[TaggedDocument]:
        """Подготовка данных для Doc2Vec"""
        documents = []
        for i, doc in enumerate(processed_corpus):
            tagged_doc = TaggedDocument(
                words=doc['processed_text'],
                tags=[f"doc_{i}"]
            )
            documents.append(tagged_doc)
        return documents
    
    def evaluate_word_analogies(self, model, analogies_file: str = None) -> Dict[str, float]:
        """Оценка точности аналогий"""
        if analogies_file and os.path.exists(analogies_file):
            return model.wv.evaluate_word_analogies(analogies_file)
        else:
            # Базовая оценка на простых аналогиях
            analogies = [
                ['москва', 'россия', 'париж', 'франция'],
                ['мужчина', 'женщина', 'король', 'королева'],
                ['хороший', 'лучше', 'плохой', 'хуже']
            ]
            
            correct = 0
            total = 0
            
            for analogy in analogies:
                try:
                    predicted = model.wv.most_similar(positive=[analogy[1], analogy[2]], 
                                                    negative=[analogy[0]], topn=1)
                    if predicted[0][0] == analogy[3]:
                        correct += 1
                    total += 1
                except:
                    continue
            
            accuracy = correct / total if total > 0 else 0
            return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def get_document_vectors(self, model, processed_corpus: List[Dict[str, Any]]) -> np.ndarray:
        """Получение векторов документов"""
        doc_vectors = []
        for i, doc in enumerate(processed_corpus):
            doc_id = f"doc_{i}"
            try:
                vector = model.dv[doc_id]
                doc_vectors.append(vector)
            except:
                # Если документа нет в модели, используем среднее векторов слов
                word_vectors = []
                for word in doc['processed_text']:
                    try:
                        word_vectors.append(model.wv[word])
                    except:
                        continue
                if word_vectors:
                    doc_vectors.append(np.mean(word_vectors, axis=0))
                else:
                    doc_vectors.append(np.zeros(self.vector_size))
        
        return np.array(doc_vectors)
    
    def compare_models(self, sentences: List[List[str]], 
                      processed_corpus: List[Dict[str, Any]]) -> pd.DataFrame:
        """Сравнение всех моделей"""
        print("Сравнение моделей распределенных представлений...")
        
        results = []
        
        # Word2Vec модели
        for sg in [0, 1]:
            model = self.train_word2vec(sentences, sg=sg)
            analogies_result = self.evaluate_word_analogies(model)
            
            results.append({
                'model': f'Word2Vec ({"Skip-gram" if sg == 1 else "CBOW"})',
                'vector_size': self.vector_size,
                'vocabulary_size': len(model.wv.key_to_index),
                'analogy_accuracy': analogies_result.get('accuracy', 0),
                'training_time': self.models[f'word2vec_{"skipgram" if sg == 1 else "cbow"}']['training_time']
            })
        
        # FastText модели (используем gensim)
        for sg in [0, 1]:
            model = self.train_fasttext(sentences, sg=sg)
            analogies_result = self.evaluate_word_analogies(model)
            
            results.append({
                'model': f'FastText ({"Skip-gram" if sg == 1 else "CBOW"})',
                'vector_size': self.vector_size,
                'vocabulary_size': len(model.wv.key_to_index),
                'analogy_accuracy': analogies_result.get('accuracy', 0),
                'training_time': self.models[f'fasttext_{"skipgram" if sg == 1 else "cbow"}']['training_time']
            })
        
        # Doc2Vec модели
        documents = self.prepare_doc2vec_data(processed_corpus)
        for dm in [0, 1]:
            model = self.train_doc2vec(documents, dm=dm)
            doc_vectors = self.get_document_vectors(model, processed_corpus)
            
            results.append({
                'model': f'Doc2Vec ({"PV-DM" if dm == 1 else "PV-DBOW"})',
                'vector_size': self.vector_size,
                'vocabulary_size': len(model.wv.key_to_index),
                'analogy_accuracy': 0,  # Doc2Vec не оценивается через аналогии
                'training_time': self.models[f'doc2vec_{"pvdm" if dm == 1 else "pvdbow"}']['training_time']
            })
        
        return pd.DataFrame(results)