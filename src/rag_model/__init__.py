"""
RAG Model Package for PersonalTrainerAI

This package contains implementations of different Retrieval-Augmented Generation (RAG)
architectures for the PersonalTrainerAI project.
"""

from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .graph_rag import GraphRAG
# Comment out the problematic import until fixed
from .raptor_rag import RAPTORRAG
from .rag_integration import RAGIntegration
from .rag_evaluation import RAGEvaluator

__all__ = [
    'NaiveRAG', 
    'AdvancedRAG', 
    'ModularRAG', 
    'GraphRAG',
    'RAPTORRAG',  # Comment this out too
    'RAGIntegration',
    'RAGEvaluator'
]
