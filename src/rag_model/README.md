"""
Updated README for RAG Model Implementation in PersonalTrainerAI

This document provides an overview of the RAG (Retrieval Augmented Generation) implementations
for the PersonalTrainerAI project, including the newly added Graph RAG and RAPTOR RAG architectures.
"""

# RAG Model Implementation for PersonalTrainerAI

This directory contains the implementation of five different RAG (Retrieval Augmented Generation) architectures for the PersonalTrainerAI project. These implementations allow the AI to retrieve relevant fitness knowledge from the vector database and generate accurate, helpful responses to user queries.

## Overview

The RAG model combines the power of large language models with retrieval from a domain-specific knowledge base. For PersonalTrainerAI, this means retrieving fitness knowledge from our Pinecone vector database and using it to generate personalized fitness advice.

We've implemented five different RAG architectures to compare their performance:

1. **Naive RAG**: A baseline implementation with direct vector similarity search
2. **Advanced RAG**: Enhanced with query expansion, sentence-window retrieval, and re-ranking
3. **Modular RAG**: A flexible system with query classification and specialized retrievers
4. **RAPTOR RAG**: Employs multi-step reasoning with iterative retrieval for complex queries

## Architecture Comparison

### Naive RAG
- Simple vector similarity search
- Direct document retrieval
- Basic prompt construction
- Good for straightforward fitness queries

### Advanced RAG
- Query expansion using LLM
- Sentence-window retrieval for better context
- Re-ranking of retrieved documents
- Dynamic context window based on relevance
- Structured prompt engineering
- Better for nuanced fitness questions

### Modular RAG
- Query classification
- Specialized retrievers for different fitness topics
- Template-based responses
- Excellent for diverse query types


### RAPTOR RAG
- Query planning and decomposition
- Iterative, multi-step retrieval
- Reasoning over retrieved information
- Self-reflection and refinement
- Structured response synthesis
- Best for complex, multi-part fitness questions

## Getting Started

### Prerequisites

- Python 3.8+
- Pinecone account with an index set up
- OpenAI API key

### Environment Setup

Create a `.env` file in the project root with:

```
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=personal-trainer-ai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements include:
```
langchain>=0.1.0
langchain-openai>=0.0.2
langchain-community>=0.0.10
langchain-core>=0.1.0
openai>=1.0.0
pinecone-client>=2.2.1
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
tqdm>=4.65.0
matplotlib>=3.7.0
pydantic>=2.0.0
networkx>=2.8.0  # For Graph RAG
```

## Usage

### Comparing RAG Implementations

To compare all five RAG implementations and determine which works best for your fitness knowledge base:

```bash
python -m src.rag_model.compare_rag_implementations --output-dir results
```

This will:
1. Run test queries through all five RAG implementations
2. Evaluate responses using multiple metrics
3. Generate comparison charts and a detailed report
4. Identify the best-performing implementation

Additional options:
```bash
python -m src.rag_model.compare_rag_implementations --help
```

### Using a Specific RAG Implementation

To use a specific RAG implementation for processing queries:

```bash
python -m src.rag_model.rag_integration --implementation [naive|advanced|modular|graph|raptor]
```

For example, to use the Graph RAG implementation:
```bash
python -m src.rag_model.rag_integration --implementation graph
```


## Evaluation Framework

The evaluation framework in `rag_evaluation.py` assesses RAG performance using these metrics:

1. **Relevance**: How well the response addresses the user's query
2. **Factual Accuracy**: Whether the information provided is correct
3. **Completeness**: Whether the response covers all important aspects
4. **Hallucination**: Whether the response contains information not in the retrieved documents
5. **Relationship Awareness**: How well the response demonstrates understanding of relationships between fitness concepts
6. **Reasoning Quality**: The quality of reasoning demonstrated in complex responses

## Integration with Data Pipeline

The RAG implementations integrate with the existing data pipeline:

1. Data is scraped from fitness sources
2. Text is processed and chunked
3. Chunks are embedded and stored in Pinecone
4. RAG retrieves relevant chunks based on user queries
5. LLM generates responses using the retrieved information

## File Structure

- `__init__.py`: Package initialization
- `naive_rag.py`: Baseline RAG implementation
- `advanced_rag.py`: Enhanced RAG with additional techniques
- `modular_rag.py`: Modular RAG with specialized retrievers
- `graph_rag.py`: Graph-based RAG using knowledge graph
- `raptor_rag.py`: RAPTOR RAG with multi-step reasoning
- `rag_evaluation.py`: Evaluation framework for comparing implementations
- `compare_rag_implementations.py`: Script to run comparisons
- `rag_integration.py`: Integration with the existing pipeline
- `rag_implementation_strategy.md`: Detailed strategy document
- `README.md`: This documentation file

## Contributing

When extending or modifying the RAG implementations:

1. Ensure all implementations follow the same interface
2. Add appropriate evaluation metrics for new techniques
3. Update the comparison script to include new implementations
4. Document any new parameters or configuration options

## Next Steps

- Optimize RAPTOR RAG's reasoning process for better performance
- Implement hybrid approaches combining the strengths of different architectures
- Develop a feedback loop to continuously improve RAG performance based on user interactions
