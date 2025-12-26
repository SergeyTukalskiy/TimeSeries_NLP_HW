import os
import sys
import streamlit as st

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from web_interface import VectorSpaceExplorer
from distributed_models import DistributedModels
from gensim.models import Word2Vec, FastText

def load_models():
    """Загрузка обученных моделей"""
    models_dir = 'models'
    models = {}
    
    try:
        # Загрузка Word2Vec моделей
        if os.path.exists(os.path.join(models_dir, 'word2vec_skipgram.model')):
            models['word2vec_skipgram'] = {
                'model': Word2Vec.load(os.path.join(models_dir, 'word2vec_skipgram.model')),
                'description': 'Word2Vec Skip-gram (100D)'
            }
        
        if os.path.exists(os.path.join(models_dir, 'word2vec_cbow.model')):
            models['word2vec_cbow'] = {
                'model': Word2Vec.load(os.path.join(models_dir, 'word2vec_cbow.model')),
                'description': 'Word2Vec CBOW (100D)'
            }
        
        # Загрузка FastText моделей (gensim)
        if os.path.exists(os.path.join(models_dir, 'fasttext_skipgram.model')):
            models['fasttext_skipgram'] = {
                'model': FastText.load(os.path.join(models_dir, 'fasttext_skipgram.model')),
                'description': 'FastText Skip-gram (100D)'
            }
        
        if os.path.exists(os.path.join(models_dir, 'fasttext_cbow.model')):
            models['fasttext_cbow'] = {
                'model': FastText.load(os.path.join(models_dir, 'fasttext_cbow.model')),
                'description': 'FastText CBOW (100D)'
            }
            
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
    
    return models

def main():
    """Запуск веб-приложения"""
    st.set_page_config(
        page_title="Анализатор векторных пространств",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Анализатор векторных пространств")
    st.markdown("""
    Веб-интерфейс для исследования семантических свойств векторных представлений слов и документов.
    
    **Функциональность:**
    - 🧮 Векторная арифметика с визуализацией промежуточных шагов
    - 📊 Семантическое сходство с графами связей
    - 📈 Анализ семантических осей и смещения моделей
    - 🎨 2D/3D визуализация с выделением семантических кластеров
    - 📋 Динамические отчеты с heatmap и статистикой
    """)
    
    # Загрузка моделей
    with st.spinner("Загрузка моделей..."):
        models = load_models()
    
    if not models:
        st.error("""
        ❌ Модели не найдены!
        
        Перед запуском веб-приложения выполните:
        ```bash
        python run_experiments.py
        ```
        Это обучит необходимые модели и сохранит их в папку `models/`.
        """)
        return
    
    st.success(f"✅ Загружено {len(models)} моделей")
    
    # Инициализация и запуск интерфейса
    explorer = VectorSpaceExplorer()
    explorer.load_models(models)
    explorer.run()

if __name__ == "__main__":
    main()