"""
MLflow RAG Metrics Configuration for PersonalTrainerAI

This module defines the metrics to be tracked in MLflow for RAG evaluation experiments.
Based on the existing evaluation framework in advanced_rag_evaluation.py.
"""

# Standard metrics categories from the existing evaluation framework
RAG_METRICS = {
    # RAGAS metrics
    "ragas_metrics": [
        "faithfulness",
        "answer_relevancy",
        "context_relevancy",
        "context_precision"
    ],
    
    # Custom fitness domain metrics
    "custom_metrics": [
        "fitness_domain_accuracy",
        "scientific_correctness",
        "practical_applicability",
        "safety_consideration"
    ],
    
    # Retrieval metrics
    "retrieval_metrics": [
        "retrieval_precision",
        "retrieval_recall"
    ],
    
    # Human evaluation metrics
    "human_eval_metrics": [
        "answer_completeness",
        "answer_conciseness",
        "answer_helpfulness"
    ]
}

# Combined metrics for overall evaluation
ALL_METRICS = (
    RAG_METRICS["ragas_metrics"] + 
    RAG_METRICS["custom_metrics"] + 
    RAG_METRICS["retrieval_metrics"] + 
    RAG_METRICS["human_eval_metrics"]
)

# Additional experiment parameters to track
EXPERIMENT_PARAMS = [
    "rag_implementation",  # Type of RAG implementation (advanced, modular, raptor)
    "embedding_model",     # Embedding model used
    "llm_model",           # LLM model used
    "chunk_size",          # Document chunk size
    "chunk_overlap",       # Document chunk overlap
    "retrieval_k",         # Number of documents retrieved
    "temperature"          # LLM temperature
]

# Define experiment tags for organization
EXPERIMENT_TAGS = {
    "project": "PersonalTrainerAI",
    "module": "RAG Evaluation",
    "framework_version": "1.0"
}
