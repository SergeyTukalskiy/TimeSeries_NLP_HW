"""
Модуль для векторизации текстов и анализа векторных пространств
"""

from .utils import TextPreprocessor, CorpusLoader
from .classical_vectorizers import ClassicalVectorizers
from .dimensionality_reduction import DimensionalityReducer
from .distributed_models import DistributedModels
from .semantic_analysis import SemanticAnalyzer
from .web_interface import VectorSpaceExplorer

__version__ = "1.0.0"
__author__ = "Vectorization Project"