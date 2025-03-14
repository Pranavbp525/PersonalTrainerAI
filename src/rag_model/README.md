# RAG Implementation for PersonalTrainerAI

This directory contains the implementation of Retrieval-Augmented Generation (RAG) components for the PersonalTrainerAI project. The RAG system enhances the AI's ability to provide personalized fitness advice by retrieving relevant information from a knowledge base of fitness content.

## Overview

The RAG implementation consists of three different architectures:

1. **Naive RAG**: A baseline implementation that directly retrieves documents based on vector similarity and passes them to an LLM.
2. **Advanced RAG**: An enhanced implementation with query expansion, sentence-window retrieval, and re-ranking.
3. **Modular RAG**: A flexible, component-based implementation with query classification, specialized retrievers, and template-based response generation.

## Components

- `rag_implementation_strategy.md`: Detailed strategy document outlining the RAG implementation approach
- `naive_rag.py`: Implementation of the baseline Naive RAG approach
- `advanced_rag.py`: Implementation of the Advanced RAG approach with enhanced retrieval and context processing
- `modular_rag.py`: Implementation of the Modular RAG approach with specialized components
- `rag_evaluation.py`: Framework for evaluating and comparing different RAG implementations
- `compare_rag_implementations.py`: Script to test and compare the RAG implementations
- `rag_integration.py`: Module for integrating RAG components with the existing data pipeline

## Usage

### Evaluating RAG Implementations

To compare the different RAG implementations:

```bash
python -m src.rag_model.compare_rag_implementations --output-dir results
```

Options:
- `--test-queries`: Path to a JSON file containing test queries
- `--output-dir`: Directory to save evaluation results (default: "results")
- `--generate-queries`: Number of additional test queries to generate
- `--implementations`: Comma-separated list of implementations to test (default: "all")

### Using the RAG System

To use the RAG system for processing queries:

```bash
python -m src.rag_model.rag_integration --implementation modular
```

Options:
- `--implementation`: RAG implementation to use (naive, advanced, or modular)
- `--update-db`: Update vector database before processing queries
- `--query`: Query to process (if not provided, interactive mode is used)

## Integration with Data Pipeline

The RAG system integrates with the existing data pipeline through the `rag_integration.py` module. This module connects to the Pinecone vector database and provides functionality to update the database and process queries using the selected RAG implementation.

## Evaluation Metrics

The evaluation framework uses the following metrics to assess RAG performance:

1. **Relevance**: How well the response addresses the query
2. **Factual Accuracy**: Correctness of information in the response
3. **Completeness**: How comprehensive the response is
4. **Hallucination**: Whether the response contains information not present in the retrieved context

## Requirements

- Python 3.8+
- LangChain
- Pinecone
- OpenAI API key (for LLM access)
- HuggingFace Transformers (for embeddings)

## Future Improvements

- Fine-tuning embedding models on fitness-specific data
- Implementing hybrid search (combining vector and keyword search)
- Adding user feedback mechanisms for continuous improvement
- Exploring domain-specific LLMs for fitness content generation
